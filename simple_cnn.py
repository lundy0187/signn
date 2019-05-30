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


ds = tf.data.Dataset.from_generator(
    dp.deepsig_dataset_generator('/home/ctriant/deepsig-dataset/2018.01/', snr=[10]), 
    (tf.float32, tf.uint8),
    ([2,1024],[]))

in_shp = ds.output_shapes[0].as_list()

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


#Set up train parameters
nb_epoch = 100     # number of epochs to train on
batch_size = 1  # training batch size
dataset = ds.shuffle(buffer_size=5000)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
# dataset = dataset.prefetch(buffer_size=batch_size)

# filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(dataset, epochs=10, steps_per_epoch=32)
# model.load_weights(filepath)
