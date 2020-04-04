from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class Model(object):
    def __init__(self, proto=False, min_delta=0.001):
        self.proto = proto

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.model = self.build_model()
        self.model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        if(proto):
            self.earlystop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='accuracy', min_delta=min_delta,
                patience=4)
        else:
            self.earlystop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='accuracy', min_delta=min_delta,
                patience=20)
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
            )

        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    @classmethod
    def build_model(self, k=4):
        def group(input, kernel_size):
            layer1 = tf.keras.layers.Conv2D(kernel_size, (3, 3), padding='same')(input)
            layer1 = tf.keras.layers.BatchNormalization()(layer1)
            layer1 = tf.keras.layers.ReLU()(layer1)
            layer1 = tf.keras.layers.Conv2D(kernel_size, (3, 3), padding='same')(layer1)
            layer2 = tf.keras.layers.Conv2D(kernel_size, (1, 1), padding='same')(input)
            block = tf.keras.layers.add([layer1, layer2])

            layer1 = tf.keras.layers.BatchNormalization()(block)
            layer1 = tf.keras.layers.ReLU()(layer1)
            layer1 = tf.keras.layers.Conv2D(kernel_size, (3, 3), padding='same')(layer1)
            layer1 = tf.keras.layers.BatchNormalization()(layer1)
            layer1 = tf.keras.layers.ReLU()(layer1)
            layer1 = tf.keras.layers.Conv2D(kernel_size, (3, 3), padding='same')(layer1)
            block = tf.keras.layers.add([layer1, block])

            layers = tf.keras.layers.BatchNormalization()(block)
            block = tf.keras.layers.ReLU()(layers)

            return block

        inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
        layers = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
        layers = tf.keras.layers.BatchNormalization()(layers)
        block = tf.keras.layers.ReLU()(layers)

        block = group(block, 16 * k)
        block = group(block, 32 * k)
        block = group(block, 64 * k)

        layers = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))(block)
        block = tf.keras.layers.Flatten()(layers)

        outputs = tf.keras.layers.Dense(10, activation='softmax')(block)

        return tf.keras.Model(inputs, outputs, name="ENET")

    def run(self, predictors, targets, val_predictors=None, val_targets=None, epochs=256, batch_size=64):
        self.datagen.fit(predictors)
        if(self.proto):
            for epoch in range(epochs):
                self.model.fit(self.datagen.flow(predictors, targets, batch_size=batch_size),
                    steps_per_epoch=len(predictors)/batch_size, epochs=1, callbacks=[self.earlystop_callback, self.tensorboard_callback])
                test_loss, test_acc = self.evaluate(val_predictors, val_targets)
                print("\nTest Accuracy:", test_acc)
        else:
            self.model.fit(self.datagen.flow(predictors, targets, batch_size=batch_size),
                steps_per_epoch=len(predictors)/batch_size, epochs=epochs, callbacks=[self.earlystop_callback, self.tensorboard_callback])
        # self.model.fit(predictors, targets, epochs=epochs, callbacks=[self.earlystop_callback, self.tensorboard_callback])

    def evaluate(self, predictors, targets, verbosity=2):
        loss, acc = self.model.evaluate(predictors, targets,
            verbose=verbosity)
        return loss, acc

    def predict(self, images):
        predictions = self.model.predict(images)
        predicts = np.array([int(np.argmax(prediction)) for prediction in predictions])
        df = pd.DataFrame(predicts, columns=['predicted'])
        df.index.name = 'Id'
        return df

    def write_predictions(self, df):
        file = "predictions-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
        df.to_csv(file, index=True, header=True)

    def save_graph(self):
        file = "model-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
        tf.keras.utils.plot_model(self.model, file, show_shapes=True)
