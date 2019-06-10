import argparse
import os
import errno
import time
import tensorflow as tf
import tensorflow.python.keras.models as models
from tensorflow.python.client import device_lib

from utils import deepsig_dataset_generator as ddg


class signn_trainer():

    def __init__(self, dataset_path, model_path, epochs, steps_per_epoch,
                 batch_size, shuffle, shuffle_buffer_size, split_ratio,
                 validation_steps):
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
        self.train_samples = int((self.split_ratio[0]*self.dataset_parser
                                 .get_total_samples()))
        self.validation_samples = int((self.split_ratio[1]*self.dataset_parser
                                       .get_total_samples()))
        self.__init_dataset()
        self.model = self.__init_model()

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
        print("Batch sizes set on datasets.")

    def __init_model(self):
        if (not os.path.isfile(self.model_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.model_path)
        model = models.load_model(self.model_path)
        model.summary()
        return model

    def train(self):
        return self.model.fit(x=self.train_dataset,
                              epochs=self.epochs,
                              steps_per_epoch=None,
                              validation_data=self.validation_dataset,
                              validation_steps=None,
                              verbose=2,
                              shuffle=False)


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
    return parser


def main(trainer=signn_trainer, args=None):
    if args is None:
        args = argument_parser().parse_args()

    t = trainer(dataset_path=args.dataset_path, model_path=args.model_path,
                epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size, shuffle=args.shuffle,
                shuffle_buffer_size=args.shuffle_buffer_size,
                split_ratio=args.split_ratio,
                validation_steps=args.validation_steps)

    t.train()
    # t.print_dataset_batch()


if __name__ == '__main__':
    main()
