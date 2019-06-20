import numpy as np
import h5py


class deepsig_dataset_generator():
    """A toolkit for parsing the Deepsig Inc. 2018.01 dataset"""

    def __init__(self, dataset_path, modulation=None, snr=None,
                 split_ratio=[0.8, 0.1, 0.1]):
        self.dataset_path = dataset_path
        self.dataset = h5py.File(dataset_path+"/GOLD_XYZ_OSC.0001_1024.hdf5",
                                 'r')
        self.__init_data()
        self.__init_modulations(modulation)
        self.__init_snr()
        self.snr_num = 26
        self.samples_per_snr_mod = 4096
        self.modarg = modulation
        self.snrarg = snr
        self.mod_classes = self.list_available_modulations()
        self.total_samples_num = self.get_total_samples()
        # TODO: Add check for the split ratio list dimension
        self.split_ratio = split_ratio
        self.train_samples = int(split_ratio[0] *
                                 self.total_samples_num/self.mods_num)
        self.valid_samples = int(split_ratio[1] *
                                 self.total_samples_num/self.mods_num)
        self.test_samples = int(split_ratio[2] *
                                self.total_samples_num/self.mods_num)

    def __init_data(self):
        self.data = self.dataset['X']

    def __init_modulations(self, modulation):
        self.modulations = self.dataset['Y']
        if modulation is not None:
            self.mods_num = len(modulation)
        else:
            self.mods_num = len(self.list_available_modulations())

    def __init_snr(self):
        self.snr = self.dataset['Z']

    def __get_index(self, occur, l):
        return [l.index(idx.upper()) for idx in occur]

    def list_selected_modulations(self):
        if self.modarg is not None:
            return [m.upper() for m in self.modarg]
        else:
            self.list_available_modulations()

    def list_available_modulations(self):
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
            return (len(self.snrarg) *
                    self.mods_num *
                    self.samples_per_snr_mod)
        elif self.snrarg is None:
            return len(self.modarg)*self.snr_num*self.samples_per_snr_mod
        else:
            return self.modulations[:, 0].size

    '''
    Test dataset generator
    '''
    def test_dataset_generator(self):

        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(
                self.train_samples+self.valid_samples,
                    self.train_samples+self.valid_samples+self.test_samples):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    yield (self.data[indx[j, i], :, :].transpose(),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])

    '''
    Validation dataset generator
    '''
    def validation_dataset_generator(self):

        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(self.train_samples,
                           self.train_samples+self.valid_samples):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    yield (self.data[indx[j, i], :, :].transpose(),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])

    '''
    Train dataset generator
    '''
    def train_dataset_generator(self):

        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        # print("Total samples: ", self.total_samples_num)
        # print("Training samples: ", self.train_samples*self.mods_num)
        # print("Validation samples: ", self.valid_samples*self.mods_num)
        # print("Testing samples: ", self.test_samples*self.mods_num)
        # print('Loss:',
        #       (self.train_samples+self.valid_samples+self.test_samples) -
        #       self.total_samples_num/self.mods_num)

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(0, self.train_samples):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    yield (self.data[indx[j, i], :, :].transpose(),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])
