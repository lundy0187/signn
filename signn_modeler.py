"""
   Copyright (C) 2019, Libre Space Foundation <https://libre.space/>

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import argparse
import tensorflow.python.keras.models as models
from tensorflow.python.keras.layers.core import (Reshape, Dense, Dropout,
                                                 Activation, Flatten)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          ZeroPadding2D)


class signn_modeler():
    """
    A class that incorporates Tensorflow and Keras for the definition of the
    model architecture.

    Attributes
    ----------
    model : string
        the enumeration that defines the selected architecture
    input_shape : int
        a list of integers that defines the input shape of the architecture
    target_num : int
        the number of targets for the selected architecture
    destination: string
        the full path to save the Keras model containing the file name
        and extension
    """
    def __init__(self, model, input_shape, target_num, destination):
        self.model_choice = model
        self.input_shape = input_shape
        self.target_num = target_num
        self.model = self.__get_model()
        self.destination = destination

    def __get_model(self):
        if (self.model_choice == "deepsig"):
            return self.__get_deepsig_cnn()

    def __get_deepsig_cnn(self):
        """
        Return the model that describes the CNN architecture as described from
        Deepsig Inc.
        """
        dr = 0.5
        model = models.Sequential()
        model.add(Reshape(self.input_shape + [1],
                          input_shape=self.input_shape))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv1",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(80, (1, 3), activation="relu", name="conv12",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name="dense1"))
        model.add(Dropout(dr))
        model.add(Dense(self.target_num, name="dense2"))
        model.add(Activation('softmax'))
        model.add(Reshape([self.target_num]))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        return model

    def export_model(self):
        """
        Save the generated architecture.
        """
        self.model.save(self.destination)

    def print_model_summary(self):
        """
        Print the summary of the generated architecture.
        """
        self.model.summary()


def argument_parser():
    description = 'A tool to generate models using Keras/Tensorflow'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-m", "--model", default='deepsig', nargs='?',
                        dest="model", action="store", choices=['deepsig'],
                        help='Choose the model to generate. \
                            (Default: %(default)s)')
    parser.add_argument("-i", "--input-shape", dest="input_shape", nargs='+',
                        type=int, help='Set the model\'s input shape',
                        required=True)
    parser.add_argument("-n", "--target-num", dest="target_num", default=24,
                        type=int, help='Set the number of target classes.')
    parser.add_argument("-s", "--save", dest="destination", action='store',
                        help="Export the generated model at the given path.")
    parser.add_argument("-v", "--verbose", dest="verbose", action='store_true',
                        help="Print info regarding the generated model.")
    return parser


def main(modeler=signn_modeler, args=None):
    if args is None:
        args = argument_parser().parse_args()

    m = modeler(model=args.model, input_shape=args.input_shape,
                target_num=args.target_num, destination=args.destination,)
    if (args.destination):
        m.export_model()
    if (args.verbose):
        m.print_model_summary()


if __name__ == '__main__':
    main()
