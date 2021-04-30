import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def plot_history(model_history, plot=True):
    '''
    Plots model history and saves it the file
    '''
    if plot:
        accuracy = model_history['accuracy']
        val_accuracy = model_history['val_accuracy']

        loss = model_history['loss']
        val_loss = model_history['val_loss']

        epochs = list(range(0, len(accuracy)))

        plt.title(f"val acc: {max(val_accuracy):.3f}")

        plt.plot(epochs, accuracy, label='accuracy')
        plt.plot(epochs, val_accuracy, label='val accuracy')

        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val loss')

        plt.grid(True)
        plt.legend()

        plt.xlabel("Epoch")
        plt.ylabel("Measure(%)")

        plt.savefig(os.path.join(root_dir, f'figures/learning_curve.png'))

        plt.show()


def generate_classification_report(image_generator, predictions):
    '''
    Generates a classfication report.
    '''
    return classification_report(y_true=image_generator.classes,
                                 y_pred=predictions)


def generate_confusion_matrix(image_generator, predictions):
    '''
    Generates confusion matrix
    '''
    return confusion_matrix(image_generator.classes, predictions)


def get_metrics(image_generator, predictions):
    '''
    Returns accuracy, sensitivity, and specificity of predictions
    '''
    conf_mat = generate_confusion_matrix(image_generator, predictions)
    total_obs = sum(sum(conf_mat))

    acc = (conf_mat[0, 0] + conf_mat[1, 1]) / total_obs
    sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    specificity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])

    return acc, sensitivity, specificity

