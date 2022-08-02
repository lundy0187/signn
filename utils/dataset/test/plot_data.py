import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

folTest = os.getcwd() + '/utils/dataset/gnuradio_sim/'
fileTest = 'SIGNN_2019_01_1024.hdf5'

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

def parse_dataset(modName, snrName, folName, fileName):
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

def plot_proto(xVec, yVec, modName, xStr):
    plt.plot(xVec, yVec)
    plt.grid(True)
    plt.ylabel(modName + xStr)

def plot_hist(yVec, modName, xStr=' Histogram'):
    plt.hist(yVec, bins='auto')
    plt.grid(True)
    plt.ylabel(modName + xStr)

def print_plots(sig1, mod1, sig2, mod2, fs=200e3):
    # establish plotting vectors
    nLen = len(sig1)
    tVec = np.linspace(0, (nLen - 1) / fs, nLen)
    fVec = np.linspace(-1, 1, nLen) * fs / 2
    # print plots
    # print amplitude displays
    plt.figure(0)
    plt.subplot(2,1,1)
    plot_proto(tVec, 20*np.log10(np.abs(sig1)), mod1, ' Amplitude')
    plt.subplot(2,1,2)
    plot_proto(tVec, 20*np.log10(np.abs(sig2)), mod2, ' Amplitude')
    # print fft displays
    plt.figure(1)
    plt.subplot(2,1,1)
    plot_proto(fVec, 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig1)))), mod1, ' FFT')
    plt.subplot(2,1,2)
    plot_proto(fVec, 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig2)))), mod2, ' FFT')
    # histogram
    plt.figure(2)
    plt.subplot(2,1,1)
    plot_hist(np.abs(sig1), mod1)
    plt.subplot(2,1,2)
    plot_hist(np.abs(sig2), mod2)
    # show figures
    plt.show()

if __name__ == "__main__":
    # modulations
    mod1 = 'WBFM'
    mod2 = 'AM-DSB'
    snr = 10
    # extract datasets
    wbfmSet = parse_dataset(mod1, snr, folTest, fileTest)
    amdsbSet = parse_dataset(mod2, snr, folTest, fileTest)
    # extract example vectors for analysis
    wbfmSig = wbfmSet[np.random.randint(wbfmSet.shape[0]), :]
    amdsbSig = amdsbSet[np.random.randint(amdsbSet.shape[0]), :]
    # print plots
    print_plots(wbfmSig, mod1, amdsbSig, mod2)

