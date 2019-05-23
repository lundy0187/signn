import numpy as np
import h5py

import os
import errno
import random
import sys
import threading



class deepsig_dataset_parser():
    """A toolkit for parsing the Deepsig Inc. 2018.01 dataset"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = h5py.File(dataset_path+"/GOLD_XYZ_OSC.0001_1024.hdf5", 'r')
        self.__init_data()
        self.__init_modulations()
        self.__init_snr()
        self.total_samples_num = self.modulations[:,0].size
        self.modulations_num = self.modulations[0,:].size
        self.snr_num = 26
        self.mod_classes = self.list_modulations()
        
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
    
    '''
    Get data from the dataset.
    '''
    def get_data(self, modulation=None, snr=None):
        ind = []
        labels = []
        if modulation is not None:
            mods = self.__get_index(modulation, self.mod_classes)
            if snr is not None:
                for column in (self.modulations[:,mods] == 1).T:
                    q = np.where(column & 
                                (self.snr[:] == snr).any(axis=1))[0]
                    s = q.shape[0]
                    ind.append(q)
                labels.append(self.modulations[q,:])
            elif snr is None:
                for column in (self.modulations[:,mods] == 1).T:
                    q= np.where(column)[0]
                    s = q.shape[0]
                    ind.append(q)
                labels.append(np.repeat(mods,s))    
        elif modulation is None:
            if snr is not None:
                q = np.where(self.snr[:] == snr)[0]
                s = q.shape[0]
                ind.append(q)
                labels.append(np.repeat([mod for mod in range(0,len(self.mod_classes)+1,1)],s))
            elif snr is None:
                labels.append(np.repeat([mod for mod in range(0,len(self.mod_classes)+1,1)],self.total_samples_num))
                return [self.data, self.modulations]
        
        indx = (np.array(ind).reshape(-1,1))
        if indx.size == 0:
            return None;
        else:
            return [self.data[indx,:,:], self.modulations[indx,:]]
                
 
