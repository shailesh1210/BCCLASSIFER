import os
import numpy as np
import pickle
from core.classifier import Classifier
from core.utils import plot_history, generate_classification_report, generate_confusion_matrix, get_metrics
from core.preprocessor import Preprocessor
from tensorflow.keras.preprocessing import image

PKG_DIR = os.path.dirname(__file__)


def generate_images_array(image_paths):
    '''Loads image from file and converts it into array'''
    test_images = list()
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(48, 48))

        # Convert into array
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Normalize tensor
        img_tensor = img_tensor / 255.0

        # Add to the list
        test_images.append(img_tensor)

    return np.vstack(test_images)


def load_model_history(history_path):
    if not os.path.exists(history_path):
        raise Exception(f'{history_path} does not exist!')

    model_history = pickle.load(open(history_path, 'rb'))

    return model_history


def load_classifier(model_path):
    '''
    Loads and returns trained CNN classifier
    '''
    c = Classifier(model_load=True)
    c.load(filepath=model_path)

    return c


def run_predict_ui(model_path, history_path, image_paths):
    '''
    Makes prediction on set of input images. This is used by the UI.
    '''
    images = generate_images_array(image_paths=image_paths)
    model_history = load_model_history(history_path=history_path)

    plot_history(model_history=model_history, plot=True)

    classifier = load_classifier(model_path=model_path)
    classifier.compile()

    return classifier.predict(images)


def run_predict_console(processor, model_path, history_path, batch_size=32):
    '''
    Makes prediction on set of input images. This is used on console mode
    '''
    model_history = load_model_history(history_path=history_path)
    plot_history(model_history=model_history, plot=True)

    test_images = processor.generate_images('test', batch_size=batch_size)
    num_test_images = len(test_images)

    classifier = load_classifier(model_path=model_path)
    classifier.compile()

    predictions = classifier.predict(test_data=test_images, steps=num_test_images, for_ui=False)

    predictions = np.argmax(predictions, axis=1)

    print(generate_classification_report(image_generator=test_images, predictions=predictions))
    print(generate_confusion_matrix(image_generator=test_images, predictions=predictions))
    print(get_metrics(image_generator=test_images, predictions=predictions))


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(PKG_DIR, '..', 'resources'))

    model_path = os.path.abspath(os.path.join(PKG_DIR, '..', 'weights/final/trained_model_32_0.0001_Adam.hdf5'))
    history_path = os.path.abspath(os.path.join(PKG_DIR, '..', 'weights/final/model_history_32_0.0001_Adam.pkl'))

    data_config = {"training": {"size": 0.8}, "test": {"size": 0.2}, "validation": {"size": 0.1}}

    batch_size = 32

    p = Preprocessor(base_path=data_path, datasets=data_config)

    run_predict_console(p, model_path, history_path)

