## signn: Signal Detection and Classification using Neural Networks

## Disclaimer
Project forked from https://gitlab.com/librespacefoundation/sdrmakerspace/signn, commit SHA hash acb41e9d8c38f959156b68e694837ab3022f49b5  

## Requirements
The requirements are listed in the *requirements3.txt* located in the root directory of this project.

* Tensorflow [2.0.0]
* Keras [2.2.4]
* sklearn
* numpy [1.16.2]
* matplotlib [3.1.0]
* h5py [2.9.0]
* keras-tuner [1.0.0]  

In addition, for running some auxilliary scripts some extra dependencies should be met:

*  GNURadio [3.7]  

## Download & Install

~~~~
$ git clone https://gitlab.com/londonium6/signn.git
$ cd signn
$ python3 -m pip install --user -r requirements3.txt
~~~~

Then download and extract the source material needed for the generation of signals:
~~~~
$ cd utils/dataset/gnuradio_sim
$ wget https://cloud.libre.space/s/rzS3QaXLY6BTN3x/download/source_material.tar.gz
$ tar xvzf source_material.tar.gz
$ mv gutenberg_shakespeare.txt ../source_material/
$ mv serial-s01-e01.wav ../source_material/
~~~~

In order to install GNURadio 3.7 please check for the appropriate developement package for your distribution.

## Docker Alternative

In contrast with the *Requirements* and *Download & Install* steps above, one can alternatively build a Docker image using the provided *Dockerfile* and then run `signn` within the container:

~~~~
$ sudo docker build --no-cache -t net-signn:latest .
$ sudo docker run -it -v ${PWD}/workspace:/root/workspace --network=host net-signn
~~~~

## Usage

##### 1. Generate the signal dataset:

Then, run the dataset generation script:
~~~~
$ cd utils/dataset/gnuradio_sim
$ python2 generate_sim_dataset.py
~~~~

This command will create the *SIGNN_2019_01.hdf5* dataset. This step can be skipped in order to use a different signal dataset, as long as it follows the specific scheme described here. Depending on the processing power of your machine, this step can take as long as several hours.  

If you desire to skip the data generation step, then you simply replace the outputs: these are the `classes.txt` and `SIGNN_2019_01.hdf5` files used in the rest of this tutorial. Copy it to the `utils/dataset/gnuradio_sim` folder.  

##### 2. Generate the Keras model used for training:

~~~~
$ python3 signn_modeler.py -i 2 1024 -s model.h5
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_modeler.py --help
~~~~

##### 3. Train the Keras model exported with the previous command:

~~~~
$ python3 signn_trainer.py -p utils/dataset/gnuradio_sim -d SIGNN_2019_01_1024.hdf5 -m model.h5 --train --dataset-shape 2 1024 --snr 28 &
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_trainer.py --help
~~~~

##### 4. Monitor the training progress using Tensorboard:

~~~~
$ tensorboard --logdir logs/plots
~~~~

Then navigate to http://localhost:6006


##### 5. Tune Keras model hyperparameters by using keras-tuner library.

~~~~
$ python3 signn_tuner.py -p utils/dataset/gnuradio_sim -d SIGNN_2019_01.hdf5 --dataset-shape 2 1024 -s artifacts --test
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_tuner.py --help
~~~~
