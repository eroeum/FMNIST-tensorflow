from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class Model(object):
    def __init__(self, proto=False, min_delta=0.001):
        self.proto = proto

        self.model = self.build_model()
        # self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=['accuracy'])
        self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        self.earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', min_delta=min_delta,
            patience=5)
        # self.datagen = ImageDataGenerator(
        #     featurewise_center=False,
        #     samplewise_center=False,
        #     featurewise_std_normalization=False,
        #     samplewise_std_normalization=False,
        #     zca_whitening=False,
        #     zca_epsilon=1e-06,
        #     rotation_range=0,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     shear_range=0.,
        #     zoom_range=0.,
        #     channel_shift_range=0.,
        #     fill_mode='nearest',
        #     cval=0.,
        #     horizontal_flip=True,
        #     vertical_flip=False,
        #     rescale=None,
        #     preprocessing_function=None,
        #     data_format=None,
        #     validation_split=0.0)
        self.datagen = ImageDataGenerator(
            # rescale=1./255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True
            )

        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    @classmethod
    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.AveragePooling2D(pool_size=(8, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def run(self, predictors, targets, epochs=80, batch_size=64):
        self.datagen.fit(predictors)
        if(self.proto):
            self.model.fit(self.datagen.flow(predictors, targets, batch_size=batch_size),
                steps_per_epoch=len(predictors)/batch_size, epochs=epochs, callbacks=[self.earlystop_callback, self.tensorboard_callback])
        else:
            self.model.fit(self.datagen.flow(predictors, targets, batch_size=batch_size),
                steps_per_epoch=len(predictors)/batch_size, epochs=epochs, callbacks=[self.tensorboard_callback])
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
