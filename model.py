import tensorflow as tf
from tensorflow import keras

class Model(object):
    def __init__(self):
        self.model = self.build_model()
        self.model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

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
        self.model.fit(predictors, targets, epochs=epochs)

    def evaluate(self, predictors, targets, verbosity=2):
        loss, acc = self.model.evaluate(predictors, targets,
            verbose=verbosity)
        return loss, acc