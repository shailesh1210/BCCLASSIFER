
import os
import numpy as np
from tensorflow.keras.optimizers import Adagrad, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

try:
    from .classifier import Classifier
    from .preprocessor import Preprocessor
except:
    from classifier import Classifier
    from preprocessor import Preprocessor

pkg_dir = os.path.dirname(__file__)


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


def early_stop(monitor='val_accuracy', min_delta=0.001, patience=5):
    return EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)


def model_checkpoint(optimizer, batch_size, monitor='val_accuracy', mode='max'):
    # Checkpoint based on validation accuracy
    if not os.path.exists(os.path.join(pkg_dir, '../weights')):
        os.makedirs(os.path.join(pkg_dir, '../weights'))

    file_path = os.path.join(pkg_dir, f'weights/weights_{optimizer}_{batch_size}.hdf5')

    return ModelCheckpoint(file_path, monitor=monitor, save_best_only=True, verbose=0, mode=mode)


def load_data(batch_size):
    '''
    Loads training, validation and test data from resources.
    '''
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources'))
    data_config = {"training": {"size": 0.8}, "test": {"size": 0.1}, "validation": {"size": 0.1}}

    p = Preprocessor(base_path=data_path, datasets=data_config)

    train_data = p.generate_images('training', shuffle=True, batch_size=batch_size)
    valid_data = p.generate_images('validation', shuffle=False, batch_size=batch_size)

    class_weights = p.get_class_weights()

    return train_data, valid_data, class_weights


def get_optimizer(name, lr, epochs):
    '''
    Returns Optimizer object based on the name of optimizer
    '''
    if name == 'Adam':
        return Adam(lr=lr, decay=lr/epochs)
    elif name == 'Adagrad':
        return Adagrad(lr=lr, decay=lr/epochs)
    elif name == 'SGD':
        return SGD(lr=lr, momentum=0.9, decay=lr/epochs)


def create_model(hyper_params, input_shape=(48, 48), model_load=False):
    '''Creates and returns CNN model'''

    pass


def train_model(hyper_params):
    '''
    Trains CNN based BCCLASSIFER model
    :param hyper_params: dictionary of hyperparameters
    :return:
    '''
    batch_size = hyper_params['batch_size']
    epochs = hyper_params['epochs']

    lr = hyper_params['lr']
    optimizer_name = hyper_params['optimizer']

    kernel_size = hyper_params['kernel_size']
    kernel_stride = hyper_params['kernel_stride']

    pool_size = hyper_params['pool_size']
    pool_stride = hyper_params['pool_stride']

    train_data, valid_data, class_weights = load_data(batch_size=batch_size)

    train_batch_size = len(train_data)
    valid_batch_size = len(valid_data)

    c = Classifier(input_shape=train_data.image_shape,
                   kernel_size=(kernel_size, kernel_size),
                   kernel_strides=(kernel_stride, kernel_stride),
                   pool_size=(pool_size, pool_size),
                   pool_strides=(pool_stride, pool_stride),
                   model_load=False)

    c.summary()

    c.compile(optimizer=get_optimizer(optimizer_name, lr, epochs))

    callbacks = [early_stop(), model_checkpoint(optimizer=optimizer_name, batch_size=batch_size)]

    c.fit(train_data=train_data,
          valid_data=valid_data,
          steps_per_epoch=train_batch_size,
          validation_steps=valid_batch_size,
          class_weights=class_weights,
          epochs=epochs,
          callbacks=callbacks)

    c.save(filepath=os.path.join(pkg_dir, '../weights'))


if __name__ == '__main__':
    hyper_params = {'kernel_size': 5, 'kernel_stride': 1,
                    'pool_size': 2, 'pool_stride': 2, 'epochs': 150,
                    'optimizer': 'Adagrad', 'lr': 0.0001, 'batch_size': 32}

    train_model(hyper_params=hyper_params)

