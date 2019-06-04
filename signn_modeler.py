import argparse
import os
import errno
import sys

from enum import Enum

import tensorflow as tf
import tensorflow.python.keras.models as models

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.regularizers import *
from tensorflow.python.keras.optimizers import adam 

        
class signn_modeler():
        
    def __init__(self, model, input_shape, destination):
        self.model_choice = model
        self.input_shape = input_shape
        self.model = self.get_model()
        self.destination = destination
        
    def get_model(self):
        if (self.model_choice == "deepsig"):
            return self.get_deepsig_cnn()
    
    def get_deepsig_cnn(self):
        dr = 0.5
        model = models.Sequential()
        model.add(Reshape([1]+self.input_shape, input_shape=self.input_shape))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv1",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv12",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name="dense1"))
        model.add(Dropout(dr))
        model.add(Dense( 24, name="dense2" ))
        model.add(Activation('softmax'))
        model.add(Reshape([24]))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
        
    def export_model(self):
        self.model.save(self.destination)
        
    def print_model_summary(self):
        self.model.summary()
        
def argument_parser():
    description = 'A tool to generate models using Keras/Tensorflow'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=description)
    parser.add_argument("-m", "--model", default='deepsig', nargs='?', dest="model",
                        action="store", choices=['deepsig'],
                        help='Choose the model to generate. (Default: %(default)s)')
    parser.add_argument("-i", "--input-shape", dest="input_shape", nargs='+',
                        type=int, help='Set the model\'s input shape', required=True)
    parser.add_argument("-s", "--save", dest="destination", action='store',
                        help="Export the generated model at the given path.")
    parser.add_argument("-v", "--verbose", dest="verbose", action='store_true',
                        help="Print info regarding the generated model.")
    return parser

def main(modeler=signn_modeler, args=None):
    if args is None:
        args = argument_parser().parse_args()

    m = modeler(model=args.model, input_shape=args.input_shape, destination=args.destination)
    if (args.destination):
        m.export_model()
    if (args.verbose):
        m.print_model_summary()

if __name__ == '__main__':
    main()
