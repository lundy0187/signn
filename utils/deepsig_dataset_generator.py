import numpy as np
import h5py

import os
import errno
import random
import sys
import threading

import tensorflow as tf



class deepsig_dataset_generator():
    """A toolkit for parsing the Deepsig Inc. 2018.01 dataset"""

    def __init__(self, dataset_path, modulation=None, snr=None):
        self.dataset_path = dataset_path
        self.dataset = h5py.File(dataset_path+"/GOLD_XYZ_OSC.0001_1024.hdf5", 'r')
        self.__init_data()
        self.__init_modulations()
        self.__init_snr()
        self.total_samples_num = self.modulations[:,0].size
        self.modulations_num = self.modulations[0,:].size
        self.snr_num = 26
        self.samples_per_snr_mod = 4096
        self.mod_classes = self.list_modulations()
        self.modarg = modulation
        self.snrarg = snr
        
    def __init_data(self):
        self.data = self.dataset['X']
    
    def __init_modulations(self):
        self.modulations = self.dataset['Y']
        
    def __init_snr(self):
        self.snr = self.dataset['Z']

    def __get_index(self, occur, l):
        return [l.index(idx.upper()) for idx in occur]
        
    def list_modulations(self):
        f = open(self.dataset_path+'/classes.txt', 'r')
        s = f.read()
        class_list = s.split("',\n '")
        class_list[0] = class_list[0].split("classes = ['")[1]
        class_list[-1] = class_list[-1].split("']\n")[0]
        return class_list
    
    def get_total_samples(self):
        if self.modarg is not None and self.snrarg is not None:
            return len(self.modarg)*len(self.snrarg)*self.samples_per_snr_mod
        elif self.modarg is None:
            return len(self.snrarg)*self.modulations_num*self.samples_per_snr_mod
        elif self.snrarg is None:
            return len(self.modarg)*self.snr_num*self.samples_per_snr_mod
        else:
            return self.total_samples_num
    
    '''
    Get data from the dataset.
    '''
    def __call__(self):
        ind = []
        labels = []
        
        if self.snrarg is None:
            self.snrarg = [i for i in range(-20,32,2)]
                
        if self.modarg is None:
            self.modarg = self.list_modulations()
        
        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column & 
                        (self.modulations[:,mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1,1))
            for i in range(0,indx.shape[0]):
                yield (self.data[indx[i],:,:][0].transpose(), np.argmax(self.modulations[indx[i],:][0]))
 
