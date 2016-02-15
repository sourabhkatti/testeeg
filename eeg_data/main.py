import eeg_data.face.face as face
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np


# [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s]
#     0      1          2             3            4       5

def getdatasets_eeg():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[2], output


def getfft(raw_data):
    raw_data = np.asarray(raw_data)

    # Sampling frequency
    fs = 128.0

    # Time sampling interval in s
    T = 1 / fs

    # Number of samples
    data_shape = raw_data.shape
    n = data_shape[0]

    # Get x values
    x = np.linspace(0.0, n * T, n)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), n / 2)

    # Plot fft of each EEG channel
    for i in range(0, data_shape[1]):
        channel_data = raw_data[:, i]
        y = channel_data
        yf = fft(y)
        plt.semilogy(xf[0: n / 2], 2.0 / n * np.abs(yf[0:n / 2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude (db)")
    plt.savefig("C:/testeeg/testeeg/mozart/logs/fft.png")

    plt.show()
    plotspectrogram(x, xf, raw_data, n, fs)

    return xf, yf


def plotspectrogram(x, xf, raw_data, nfft, fs):
    data_shape = raw_data.shape

    print("Generating spectrogram...")
    f_lim_Hz = [0, 50]  # frequency limits for plotting
    plt.figure(figsize=(10, 5))
    for i in range(12, data_shape[1]):
        plt.clf()
        ax1 = plt.subplot(211)



        plt.subplot(212, sharex=ax1)
        f, t, Sxx = signal.spectrogram(x=raw_data[:, i], nfft=nfft, fs=fs, noverlap=100)  # returns PSD power per Hz
        plt.pcolormesh(t, f, Sxx)


        plt.clim([-25, 26])
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')

        plt.show()


def getdatasets_blinkonce():
    eeg_input, output = face.get_blink_once_ds()
    return eeg_input[2], output


def getdatasets_gyroxy():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[0:1], output


def getdatasets_samplenums():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[3], output


def getdatasets_timems():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[4], output


def getdatasets_times():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[5], output
