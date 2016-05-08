import eeg_data.face.face as face
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import time


# [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s]
#     0      1          2             3            4       5

def getdatasets_eeg():
    eeg_input, output = face.get_blink_twice_ds()
    return eeg_input[2], output


def getdatasets_eyes_open():
    eeg_input = face.get_eyes_open_ds()
    return eeg_input[2]


def getdatasets_eyes_closed():
    eeg_input = face.get_eyes_closed_ds()
    return eeg_input[2]


def getdatasets_test_eye_states():
    eeg_input = face.get_eye_states_test_ds()
    return eeg_input


def getdatasets_blink_ten():
    eeg_input, output = face.get_blink_ten_ds()
    return eeg_input[2], output


def getfft(raw_data_all):
    # Define the channels to get from the 14 channel raw data
    channel_bottom = 5
    channel_top = 13
    raw_data = np.asarray(raw_data_all[:, channel_bottom:channel_top])

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
    fft_data = []

    # Plot fft of each EEG channel
    for i in range(0, data_shape[1]):
        channel_data = raw_data[:, i]
        y = channel_data
        yf = fft(y)
        plt.semilogy(xf[0: n / 2], 2.0 / n * np.abs(yf[0:n / 2]))
        fft_data.append(yf)

    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude (db)")
    plt.savefig("C:/Users/SourabhKatti/Documents/engine/mozart/logs/fft.png")

    plt.show()
    spectro_data = plot_spectrogram(raw_data, n, fs, channel_bottom + 1, batchsize=0)
    csd_data = plot_csd(raw_data, n, fs, channel_bottom + 1)

    return xf, fft_data, spectro_data, csd_data


def plot_spectrogram(raw_data, nfft, fs, channel_bottom, batchsize=0):
    data_shape = raw_data.shape

    print("Generating spectrogram...")
    plt_num = 1
    plt.clf()
    plt.figure(1)

    channel_data = []
    # Get spectrogram for each channel
    if batchsize == 0:
        for i in range(0, data_shape[1]):
            plt.subplot(4, 2, plt_num)

            f, t, Sxx = signal.spectrogram(x=raw_data[:, i], fs=256, noverlap=127, nperseg=128,
                                           scaling='density')  # returns PSD power per Hz
            plt.pcolormesh(t, f, Sxx)

            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Channel %s' % (i + channel_bottom))
            plt_num += 1
            channel_data.append([f, t, Sxx])

        plt.show()
    return channel_data


def plot_csd(raw_data, nfft, fs, channel_bottom):
    data_shape = raw_data.shape
    channel_data = []

    print("Generating cross spectral density graph...")
    plt_num = 1
    plt.clf()
    plt.figure(1)
    for i in range(0, data_shape[1] - 1):
        plt.subplot(4, 2, plt_num)
        x = raw_data[:, i]
        y = raw_data[:, i + 1]
        f, Pxy = signal.csd(x, y, nfft=nfft, fs=fs)  # returns PSD power per Hz
        plt.semilogy(f, np.abs(Pxy))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('CSD [V**2/Hz]')
        plt.title('Channel %s' % (i + channel_bottom))
        plt_num += 1
        channel_data.append([f, Pxy])

    plt.show()
    return channel_data


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
