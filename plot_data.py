import numpy as np
import matplotlib.pyplot as plt
import h5py

folName = './utils/dataset/gnuradio_sim/'
fileName = 'SIGNN_2019_01_1024.hdf5'

#mod dictionary
modDict = {
    'BPSK': 1, 
    'QPSK': 2, 
    '8PSK': 3, 
    'QAM16': 4, 
    'QAM64': 5, 
    'GFSK': 6, 
    'CPFSK': 7,
    'WBFM': 8, 
    'AM-DSB': 9,
    'AM-SSB': 10, 
    'NOISE': 11
    }

def parse_dataset(modName, snrName):
    # import file
    dataset = h5py.File(folName + fileName, 'r')
    # import datasets
    xSet = dataset['X']
    ySet = dataset['Y']
    zSet = dataset['Z']
    # identify matched snr values
    snrInd = (zSet[:, 0] == snrName)
    # identify matched modulations
    modInd = (ySet[:, modDict[modName]] == 1)
    # produce desired vector indicies
    outInd = snrInd & modInd
    # combine I and Q channels into single complex output
    return xSet[outInd, :, 0] + 1j * xSet[outInd, :, 1]

def print_plots(sig1, mod1, sig2, mod2):
        # print plots
        # print amplitude displays
        plt.figure(0)
        plt.subplot(2,1,1)
        plt.plot(20*np.log10(np.abs(sig1)))
        plt.xlabel(mod1 + ' Amplitude')
        plt.subplot(2,1,2)
        plt.plot(20*np.log10(np.abs(sig2)))
        plt.xlabel(mod2 + ' Amplitude')
        # print fft displays
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig1)))))
        plt.xlabel(mod1 + ' FFT')
        plt.subplot(2,1,2)
        plt.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig2)))))
        plt.xlabel(mod2 + ' FFT')
        # instantaneous frequency
        plt.figure(2)
        plt.subplot(2,1,1)
        plt.plot(np.diff(np.unwrap(np.angle(sig1))))
        plt.xlabel(mod1 + ' Phase Difference')
        plt.subplot(2,1,2)
        plt.plot(np.diff(np.unwrap(np.angle(sig2))))
        plt.xlabel(mod2 + ' Phase Difference')
        # show figures
        plt.show()

if __name__ == "__main__":
    # modulations
    mod1 = 'WBFM'
    mod2 = 'AM-DSB'
    snr = 10
    # extract datasets
    wbfmSet = parse_dataset(mod1, snr)
    amdsbSet = parse_dataset(mod2, snr)
    # extract example vectors for analysis
    wbfmSig = wbfmSet[np.random.randint(wbfmSet.shape[0]), :]
    amdsbSig = amdsbSet[np.random.randint(amdsbSet.shape[0]), :]
    # print plots
    print_plots(wbfmSig, mod1, amdsbSig, mod2)

