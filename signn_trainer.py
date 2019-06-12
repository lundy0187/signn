import argparse
import os
import errno
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.python.keras.models as models
import tensorflow.python.keras.callbacks as clbck
from tensorflow.python.client import device_lib
from datetime import datetime

from utils import deepsig_dataset_generator as ddg


class signn_trainer():

    def __init__(self, dataset_path, model_path, epochs, steps_per_epoch,
                 batch_size, shuffle, shuffle_buffer_size, split_ratio,
                 validation_steps, artifacts_dest):
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.split_ratio = split_ratio
        self.validation_steps = validation_steps
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.dataset_parser = ddg.deepsig_dataset_generator(
            self.dataset_path, snr=[20], split_ratio=split_ratio)
        self.train_samples_num = int((self.split_ratio[0]*self.dataset_parser
                                      .get_total_samples()))
        self.validation_samples_num = int((self.split_ratio[1] *
                                           self.dataset_parser
                                           .get_total_samples()))
        self.test_samples_num = int((self.split_ratio[2] *
                                     self.dataset_parser.get_total_samples()))
        self.__init_dataset()
        self.model = self.__init_model()
        self.artifacts_dest = artifacts_dest
        self.logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(self.logdir)

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def __init_dataset(self):
        if (not os.path.exists(self.dataset_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.dataset_path)
        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.train_dataset_generator(),
            (tf.float32, tf.uint8), ([2, 1024], []))
        print("Train dataset initialization done.")
        self.validation_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.validation_dataset_generator(),
            (tf.float32, tf.uint8), ([2, 1024], []))
        print("Validation dataset initialization done.")
        self.test_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.test_dataset_generator(),
            (tf.float32, tf.uint8), ([2, 1024], []))
        print("Test dataset initialization done.")
        if self.shuffle:
            self.train_dataset = self.train_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=int(round(time.time() * 1000)),
                reshuffle_each_iteration=False)
            self.validation_dataset = self.validation_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=int(round(time.time() * 1000)),
                reshuffle_each_iteration=False)
            print("Shuffling datasets on.")
        self.train_dataset = self.train_dataset.batch(
            batch_size=self.batch_size)
        self.validation_dataset = self.validation_dataset.batch(
            batch_size=self.batch_size)
        self.test_dataset = self.test_dataset.batch(
            batch_size=self.batch_size)
        print("Batch sizes set on datasets.")

    def __init_model(self):
        if (not os.path.isfile(self.model_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.model_path)
        model = models.load_model(self.model_path)
        model.summary()
        return model

    def __generate_plots(self):
        if (not os.path.exists(self.artifacts_dest)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.artifacts_dest)
        plt.figure()
        plt.title('Training Performance')
        plt.plot(self.history.epoch, self.history.history['loss'],
                 label='Train Loss + Error')
        plt.plot(self.history.epoch, self.history.history['val_loss'],
                 label='Validation Error')
        plt.legend()
        plt.savefig(self.artifacts_dest+'/training_history.png',
                    bbox_inches='tight')

    def __plot_confusion_matrix(self, cm, title='Confusion matrix',
                                cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.artifacts_dest+'/confussion.png',
                    bbox_inches='tight')

    def evaluate(self):
        score = self.model.evaluate(self.test_dataset,
                                    workers=16,
                                    use_multiprocessing=True)
        print(score)
        return score

    def predict(self):
        # self.model = models.load_model(
        #     "/home/ctriant/projects/signn/artifacts/trained_model.h5")
        predictions = self.model.predict(self.test_dataset)
        truth_labels = np.array([])
        for i in self.test_dataset:
            truth_labels = np.concatenate((truth_labels, np.array(i[1])),
                                          axis=None)
        matr = tf.math.confusion_matrix(
            truth_labels,
            np.argmax(predictions, axis=1),
            num_classes=None,
            dtype=tf.dtypes.int32,
            name=None,
            weights=None
        )
        self.__plot_confusion_matrix(matr)

    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def train(self):
        filepath = self.artifacts_dest+"/trained_model.h5"
        callback_list = [
            clbck.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                  save_best_only=True, mode='auto'),
            clbck.EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                mode='auto')]
        self.history = self.model.fit(x=self.train_dataset,
                                      epochs=self.epochs,
                                      steps_per_epoch=None,
                                      validation_data=self.validation_dataset,
                                      validation_steps=None,
                                      verbose=2,
                                      shuffle=False,
                                      workers=16,
                                      use_multiprocessing=True,
                                      callbacks=callback_list)
        self.model.load_weights(filepath)


def argument_parser():
    description = 'A tool to train a CNN using Keras/Tensorflow'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-d --dataset-path", dest="dataset_path",
                        action="store", help="Set dataset path.")
    parser.add_argument("-m --model-path", dest="model_path", action="store",
                        default="", help="Set model path.")
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=10, help="Set batch size.")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10,
                        help="Set training epochs.")
    parser.add_argument("--steps-per-epoch", dest="steps_per_epoch", type=int,
                        default=None, help="Set training steps per epoch.")
    parser.add_argument('--shuffle', dest="shuffle", action='store_true',
                        help="Shuffle the dataset.")
    parser.add_argument('--no-shuffle', dest="shuffle", action='store_false',
                        help="Do not shuffle the dataset.")
    parser.add_argument("--shuffle-buffer-size", dest="shuffle_buffer_size",
                        type=int, default=100000,
                        help="Set shuffle buffer size.")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--split-ratio", default="0.8/0.2", nargs='+',
                        dest="split_ratio", action="store", type=float,
                        help='Set the train/validation portions. \
                            (Default: %(default)s)')
    parser.add_argument("--validation-steps", dest="validation_steps",
                        type=int, default=None,
                        help="Set the number of validation steps.")
    parser.add_argument("--artifacts-directory", dest="artifacts_dest",
                        default="artifacts",
                        help="Set the destiantion folder of the training\
                            artifacts.")
    return parser


def main(trainer=signn_trainer, args=None):
    if args is None:
        args = argument_parser().parse_args()

    t = trainer(dataset_path=args.dataset_path, model_path=args.model_path,
                epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size, shuffle=args.shuffle,
                shuffle_buffer_size=args.shuffle_buffer_size,
                split_ratio=args.split_ratio,
                validation_steps=args.validation_steps,
                artifacts_dest=args.artifacts_dest)

    t.train()
    t.predict()
    # t.__generate_plots()

    # t.print_dataset_batch()


if __name__ == '__main__':
    main()
