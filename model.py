import tensorflow as tf
from tensorflow import keras

class Model(object):
    def __init__(self, proto=False, min_delta=0.001):
        self.proto = proto
        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        self.earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', min_delta=min_delta,
            patience=1)

    @classmethod
    def build_model(self):
        model = keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (4, 4), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Dropout(0.1),
            keras.layers.Conv2D(64, (4, 4), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10)
        ])
        return model

    def run(self, predictors, targets, epochs=10):
        if(self.proto):
            self.model.fit(predictors, targets,
                epochs=epochs, callbacks=[self.earlystop_callback])
        else:
            self.model.fit(predictors, targets, epochs=epochs)

    def evaluate(self, predictors, targets, verbosity=2):
        loss, acc = self.model.evaluate(predictors, targets,
            verbose=verbosity)
        return loss, acc
