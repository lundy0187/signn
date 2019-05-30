from optparse import OptionParser
import tensorflow as tf
import keras.callbacks
import tensorflow.contrib.eager as tfe

from keras.utils import np_utils
import tensorflow.python.keras.models as models
from tensorflow.python.keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.regularizers import *
from tensorflow.python.keras.optimizers import adam

from utils import deepsig_dataset_generator as dp


class signn_trainer():
    
    def __init__(self, dataset_path, epochs, steps_per_epoch):
        self.dataset_path = dataset_path
        self.dataset = self.__init_dataset()
        self.model = self.__init_model()
        #self.epochs = epochs
        #self.steps_per_epoch = steps_per_epoch

    def __init_dataset(self):
        ds = tf.data.Dataset.from_generator(
            dp.deepsig_dataset_generator('/home/ctriant/deepsig-dataset/2018.01/', snr=[10]), 
            (tf.float32, tf.uint8),
            ([2,1024],[]))
        dataset = ds.shuffle(buffer_size=5000)
        dataset = dataset.batch(batch_size=1)
        dataset = dataset.repeat()
        return dataset
    
    def __init_model(self):
        print(self.dataset)
        in_shp = (self.dataset).output_shapes[0].as_list()
        dr = 0.5
        model = models.Sequential()
        model.add(Reshape([1]+in_shp, input_shape=in_shp))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv1", padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv12", padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name="dense1"))
        model.add(Dropout(dr))
        model.add(Dense( 24, name="dense2" ))
        model.add(Activation('softmax'))
        model.add(Reshape([24]))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        return model
    
    def train(self):
        return self.model.fit(self.dataset, epochs=10, steps_per_epoch=32)


    #Set up train parameters
    #nb_epoch = 100     # number of epochs to train on
    #batch_size = 1  # training batch size
    #dataset = ds.shuffle(buffer_size=5000)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.repeat()
    # dataset = dataset.prefetch(buffer_size=batch_size)

    # filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
    #history = model.fit(dataset, epochs=10, steps_per_epoch=32)
    # model.load_weights(filepath)
    

def argument_parser():
    description = 'A tool to train a CNN using Keras/Tensorflow'
    parser = OptionParser(usage="%prog: [options]", description=description)
    parser.add_option(
        "-d", "--dataset-path", dest="dataset_path", type="string", default='',
        help="Set dataset path [default=%default]")
    parser.add_option(
        "--epochs", dest="epochs", type="int", default=10,
        help="Set training epochs [default=%default]")
    parser.add_option(
        "--steps-per-epoch", dest="steps_per_epoch", type="int", default=32,
        help="Set training steps per epoch [default=%default]")
    return parser


def main(trainer=signn_trainer, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    t = trainer(dataset_path=options.dataset_path, epochs=options.epochs,
                steps_per_epoch=options.steps_per_epoch)
    
    #t.train()

if __name__ == '__main__':
    main()

    
