import argparse
import os
import errno
import time
import tensorflow as tf
import tensorflow.python.keras.models as models

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
        self.dataset_parser = ddg.deepsig_dataset_generator(self.dataset_path,
                                                            snr=[10])
        self.train_samples = int((self.split_ratio[0]*self.dataset_parser
                                 .get_total_samples()/self.batch_size))
        self.validation_samples = int((self.split_ratio[1]*self.dataset_parser
                                       .get_total_samples()/self.batch_size))
        self.__init_dataset()
        self.model = self.__init_model()

    def __init_dataset(self):
        # TODO: Fix the dataset initialization
        if (not os.path.exists(self.dataset_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.dataset_path)
        dataset = tf.data.Dataset.from_generator(
            self.dataset_parser,
            (tf.float32, tf.uint8), ([2, 1024], []))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size,
                                      seed=int(round(time.time() * 1000)),
                                      reshuffle_each_iteration=False)
        dataset = dataset.batch(batch_size=self.batch_size)
        self.validation_dataset = dataset.take(self.validation_samples)
        self.validation_dataset = self.validation_dataset.prefetch(
            buffer_size=self.batch_size)
        self.train_dataset = dataset.skip(self.validation_samples)
        self.train_dataset = self.train_dataset.prefetch(
            buffer_size=self.batch_size)

    def __init_model(self):
        if (not os.path.isfile(self.model_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.model_path)
        model = models.load_model(self.model_path)
        model.summary()
        return model

    def __prepare_training(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(
                buffer_size=self.shuffle_buffer_size)

    def train(self):
        return self.model.fit(x=self.train_dataset,
                              epochs=self.epochs,
                              steps_per_epoch=self.steps_per_epoch,
                              validation_data=self.validation_dataset,
                              validation_steps=self.validation_steps,
                              verbose=2,
                              shuffle=False)

    def print_dataset_batch(self):
        for iter in self.train_dataset:
            print(iter)


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
                        default=32, help="Set training steps per epoch.")
    parser.add_argument('--shuffle', dest="shuffle", action='store_true',
                        help="Shuffle the dataset.")
    parser.add_argument('--no-shuffle', dest="shuffle", action='store_false',
                        help="Do not shuffle the dataset.")
    parser.add_argument("--shuffle-buffer-size", dest="shuffle_buffer_size",
                        type=int, default=10000,
                        help="Set shuffle buffer size.")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--split-ratio", default="0.8/0.2", nargs='+',
                        dest="split_ratio", action="store", type=float,
                        help='Set the train/validation portions. \
                            (Default: %(default)s)')
    parser.add_argument("--validation-steps", dest="validation_steps",
                        type=int, default=2,
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


if __name__ == '__main__':
    main()
