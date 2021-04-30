import os
import pickle
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Conv2D, MaxPooling2D, AveragePooling2D,
                                     Flatten, SeparableConv2D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam, Adagrad

PKG_DIR = os.path.dirname(__file__)


class Classifier:
    NUM_CLASSES = 2
    CLASSES = {0: "Negative", 1: "Positive"}

    def __init__(self, input_shape=(48, 48, 3), kernel_size=(5, 5), kernel_strides=(1, 1),
                 pool_size=(2, 2), pool_strides=(2, 2), model_load=False):
        self.history = None

        if not model_load:
            self.model = Sequential()

            self.input_shape = list(input_shape)

            self.kernel_size = kernel_size
            self.kernel_strides = kernel_strides

            self.pool_size = pool_size
            self.pool_strides = pool_strides

            self._add_layers()

    def _add_layers(self):
        '''
        Adds layers to the CNN classifier.
        :return:
        '''
        '''
                Adds layers to the CNN classifier.
                :return:
                '''
        # First Convolutional layer + Max pooling
        # self.model.add(ZeroPadding2D(padding=self.zero_padding, input_shape=self.input_shape))
        self.model.add(Conv2D(filters=64, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', input_shape=self.input_shape, activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=64, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', input_shape=self.input_shape, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=self.pool_size, strides=self.pool_strides))
        self.model.add(Dropout(rate=0.2))

        # Second convolution layer + Avg pooling
        # self.model.add(ZeroPadding2D(padding=self.zero_padding))
        self.model.add(Conv2D(filters=128, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=128, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=128, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=self.pool_size, strides=self.pool_strides))
        self.model.add(Dropout(rate=0.2))

        # Third convolution layer + Avg pooling
        # self.model.add(ZeroPadding2D(padding=self.zero_padding))
        self.model.add(Conv2D(filters=256, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=256, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(filters=256, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=256, kernel_size=self.kernel_size, strides=self.kernel_strides,
                              padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=self.pool_size, strides=self.pool_strides))
        self.model.add(Dropout(rate=0.2))

        # Fully connected layer
        self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.3))

        self.model.add(Dense(Classifier.NUM_CLASSES, activation='softmax'))

    def summary(self):
        '''
        Displays model's summary
        :return:
        '''
        self.model.summary()

    def save(self, filepath, batch_size, lr, opt):
        '''
        Saves the model in hdf5 format
        :return:
        '''
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.model.save(filepath=os.path.join(filepath, f'trained_model_{batch_size}_{lr}_{opt}.hdf5'))

        self.save_history(batch_size=batch_size, lr=lr, opt=opt)

    def save_history(self, batch_size, lr, opt):
        pickle.dump(self.history.history, open(f'model_history_{batch_size}_{lr}_{opt}.pkl', 'wb'))

    def load(self, filepath):
        '''Loads model from the file'''
        self.model = load_model(filepath)

    def compile(self, optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']):
        '''
        Compiles the model
        :param optimizer: SGD by default
        :param loss: Loss function, binary crossentropy by default
        :param metrics: Accuracy by default
        :return:
        '''
        print("[INFO] Compiling the model..")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_data, valid_data, steps_per_epoch, validation_steps, class_weights, epochs, callbacks):
        '''
        Trains the model
        :param train_data: Training set
        :param valid_data: Validation set
        :param steps_per_epoch: Training images to be processed per epoch
        :param validation_steps: Validation images to be processed per epoch
        :param class_weights: Class weights to account for imbalance
        :param epochs: Number of training iterations
        :param callbacks: Early stopping, and learning rate scheduler call backs
        :return:
        '''
        print("[INFO] Training the model")
        self.history = self.model.fit(x=train_data, validation_data=valid_data,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_steps=validation_steps,
                                      class_weight=class_weights,
                                      epochs=epochs,
                                      callbacks=callbacks)

    def predict(self, test_data, steps=None, for_ui=True):
        print('[INFO] Predicting..')
        if for_ui:
            pred_indexes = self.model.predict_classes(test_data)
            return [Classifier.CLASSES[idx] for idx in pred_indexes]
        else:
            return self.model.predict(x=test_data, steps=steps)
