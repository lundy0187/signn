import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import errno


class plotter():

    def __init__(self, dest_path):
        self.__init_destination_path(dest_path)

    def __init_destination_path(self, dest_path):
        if (not os.path.exists(dest_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    dest_path)
        self.dest_path = os.path.join(dest_path, '')

    def plot_confusion_matrix(self, cm, class_names, filename):

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                       decimals=2)
        plt.figure(figsize=(20, 20))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar(im)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.dest_path+filename,
                    bbox_inches='tight')

    def plot_training_validation_loss(self, history, filename):
        plt.figure()
        plt.title('Training Performance')
        plt.plot(history.epoch, history.history['loss'],
                 label='Train Loss + Error')
        plt.plot(history.epoch, history.history['val_loss'],
                 label='Validation Error')
        plt.legend()
        plt.savefig(self.dest_path+filename,
                    bbox_inches='tight')

