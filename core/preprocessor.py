import os
import random
import shutil
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

PKG_DIR = os.path.dirname(__file__)


class Preprocessor:
    RAW_DATA = "raw_data"
    SEED = 54321

    def __init__(self, base_path, datasets):
        '''
        Sets base path of the dataset, and initializes
        list of histopathology image paths
        :param data_base_path:
        '''

        self.base_path = base_path
        self.raw_data_path = os.path.join(base_path, Preprocessor.RAW_DATA)

        if not os.path.exists(self.raw_data_path):
            raise ValueError(f"Error: {self.raw_data_path} does not exist!")

        self.datasets = datasets
        self.class_weights = dict()

        self.image_paths = self._read_image_paths(path=self.raw_data_path)

        self._copy_images()
        self._compute_class_weights()

        self._create_image_generator()

    def get_image_paths(self):
        '''
        Returns the list of paths of images
        :return:
        '''
        return self.image_paths

    def get_image_count(self, data_type):
        '''
        Returns total number of images by the type
        of dataset (training, validation or test set)
        :param data_type:
        :return:
        '''
        if not data_type in self.datasets:
            raise ValueError(f'Error: {data_type} not found!')
        return len(self.datasets[data_type]['image_paths'])

    def generate_images(self, data_type, image_size=(48, 48), batch_size=32, shuffle=False):
        '''
        Generates stream of images depending on the data type (training, test or validation)
        :param data_type: training, test or validation
        :param image_size: (48 X 48) by default
        :param batch_size: 32 by default
        :param shuffle: False by default
        :return:
        '''
        if not data_type in self.datasets:
            raise ValueError(f'Error {data_type} not found!')

        print(f"[INFO] Generating {data_type.capitalize()} data")

        image_directory = os.path.join(self.base_path, data_type)
        image_gen = self.datasets[data_type]['image_generator']

        return image_gen.flow_from_directory(directory=image_directory,
                                             target_size=image_size,
                                             color_mode='rgb',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    def get_class_weights(self):
        '''
        Returns class weights
        '''
        return self.class_weights

    def _compute_class_weights(self):
        '''
        Computes the weights of training class labels
        to account for +ve/-ve cases imbalance
        :return:
        '''
        class_labels = list()

        for training_image_path in self.datasets['training']['image_paths']:
            class_label = training_image_path.split(os.path.sep)[-2]
            class_labels.append(int(class_label))

        class_labels = to_categorical(class_labels)
        class_totals = class_labels.sum(axis=0)

        for idx in range(len(class_totals)):
            self.class_weights[idx] = class_totals.max() / class_totals[idx]

    def _copy_images(self):
        '''
        Creates training, testing and validation folders,
        and copies images into respective folders based on
        split indices.
        :return:
        '''
        random.seed(Preprocessor.SEED)
        random.shuffle(self.image_paths)

        self._split_data()

        # Copy images to their respective folders
        for data_type in self.datasets:
            print(f"[INFO] Creating {data_type} data..")

            if os.path.exists(os.path.join(self.base_path, data_type)):
                print(f"[INFO] {data_type.capitalize()} exists, Skipping directory creation!")
                continue

            num_images = len(self.datasets[data_type]['image_paths'])

            for file_id, file_path in enumerate(self.datasets[data_type]['image_paths']):

                if file_id % 1000 == 0:
                    print(f'{file_id} of {num_images} for {data_type} is copied!')

                path_tokens = file_path.split(os.path.sep)

                file_name = path_tokens[-1]
                class_label = path_tokens[-2]

                target_path = os.path.join(self.base_path, data_type, class_label)
                target_file_name = os.path.join(target_path, file_name)

                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                shutil.copy2(file_path, target_file_name)

    def _split_data(self):
        '''
        Splits the data into train, test and validation sets
        :return:
        '''

        total_images = len(self.image_paths)
        for data_type in self.datasets:
            if data_type == 'training':
                split_index = round(self.datasets[data_type]['size'] * total_images)

                training_images = self.image_paths[:split_index]
                test_images = self.image_paths[split_index:]

                self.datasets[data_type].update({'image_paths': training_images})
                self.datasets.update({'test': {'image_paths': test_images}})

            elif data_type == 'validation':
                if not 'training' in self.datasets:
                    raise Exception('Error: Training type does not exist!')

                training_images = self.datasets['training']['image_paths']
                total_train_images = len(training_images)
                split_index = round(self.datasets[data_type]['size'] * total_train_images)

                validation_images = training_images[:split_index]
                training_images = training_images[split_index:]

                self.datasets[data_type].update({'image_paths': validation_images})
                self.datasets['training'].update({'image_paths': training_images})

    def _create_image_generator(self):
        '''
        Initializes images generator for training, test
        and validation set
        :return:
        '''
        for data_type in self.datasets:
            if data_type == 'training':
                self.datasets[data_type].update({'image_generator': ImageDataGenerator(rescale=1/255.0,
                                                                                       width_shift_range=0.1,
                                                                                       height_shift_range=0.1,
                                                                                       shear_range=0.01,
                                                                                       zoom_range=0.01,
                                                                                       rotation_range=10,
                                                                                       horizontal_flip=True,
                                                                                       vertical_flip=True)})
            else:
                self.datasets[data_type].update({'image_generator': ImageDataGenerator(rescale=1/255.0)})

    def _read_image_paths(self, path):
        '''
        Recursively reads the image paths and
        adds them to the list
        :param path: File path
        :return:
        '''
        image_paths = list()
        for root, _, files in os.walk(path, topdown=False):
            for file in files:
                image_paths.append(os.path.join(root, file))

        return image_paths


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources'))
    data_config = {"training": {"size": 0.8}, "test": {"size": 0.1}, "validation": {"size": 0.1}}

    batch_size = 32

    p = Preprocessor(base_path=data_path, datasets=data_config)

    train_data = p.generate_images('training', shuffle=True, batch_size=batch_size)
    valid_data = p.generate_images('validation', shuffle=False, batch_size=batch_size)



