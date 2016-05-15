import eeg_data.face.face as face
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import time
import pickle
import os


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


def pickle_freq_data(fft_time_raw, spectro_raw, csd_raw, timestamp,
                     path_to_save="C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/X_eeg_fft/"):
    fft_string = path_to_save + timestamp + "-fft.p"
    spectro_string = path_to_save + timestamp + "-spectro.p"
    csd_string = path_to_save + timestamp + "-csd.p"

    try:
        with open(fft_string, 'wb') as file:
            pickle.dump(fft_time_raw, file)
        with open(spectro_string, 'wb') as file:
            pickle.dump(spectro_raw, file)
        with open(csd_string, 'wb') as file:
            pickle.dump(csd_raw, file)
        return
    except:
        return -1


def pickle_time_data(time_data_raw, timestamp,
                     path_to_save="C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/X_eeg_fft/"):
    time_string = path_to_save + str(timestamp) + "-time.p"
    try:
        with open(time_string, 'wb') as file:
            pickle.dump(time_data_raw, file)
        return
    except:
        return -1


def load_pickled_freq_data(path_to_load="C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/X_eeg_fft/",
                           timeonly=False):
    freq_files = os.listdir(path_to_load)
    num_files = freq_files.__len__()

    latest_frequencies = freq_files[0:4]
    fft_raw = -1
    spectro_raw = 0
    csd_raw = 0
    time_raw = 0
    try:
        for file in latest_frequencies:
            freq_types = file.split("-")
            pt = path_to_load + file

            if freq_types[1] == "fft.p":
                try:
                    with open(pt, 'rb') as file1:
                        fft_raw = pickle.load(file1)
                except Exception as e:
                    print(e)
                    continue

            if freq_types[1] == "time.p":
                try:
                    with open(pt, 'rb') as file4:
                        time_raw = pickle.load(file4)
                    if timeonly:
                        break
                except Exception as e:
                    print(e)
                    continue
            elif freq_types[1] == "spectro.p":
                try:
                    with open(pt, 'rb') as file2:
                        spectro_raw = pickle.load(file2)
                except Exception as e:
                    print(e)
                    continue

            elif str(freq_types[1]) == 'csd.p':
                try:
                    with open(pt, 'rb') as file3:
                        csd_raw = pickle.load(file3)
                except Exception as e:
                    print(e)
                    continue
    except:
        return fft_raw, spectro_raw, csd_raw, time_raw
    return fft_raw, spectro_raw, csd_raw, time_raw


def streamfft(yf, eeg_data, batch_size):
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude (db)")
    # plt.savefig("C:/testeeg/testeeg/mozart/logs/fft.png")


    # plt.ion()
    fft_size = np.shape(yf)
    T = 1 / 128.0
    end_x = (fft_size[1] + batch_size) / 128.0
    xeeg = np.linspace(0.0, end_x, num=(fft_size[1] + batch_size))

    for channel_num in range(0, fft_size[0]):
        print("Streaming channel %d" % (channel_num + 1))
        for batch_num in range(0, fft_size[1]):
            current_batch_fft = np.asarray(yf[channel_num][batch_num])
            current_batch_eeg = np.asarray(eeg_data[channel_num][batch_num])
            getfreqmag_batch(current_batch_fft, T)
            xf = np.linspace(0.0, 1.0 / (2.0 * T), batch_size / 2)
            xeeg_batch = xeeg[batch_num: batch_num + 256]
            index_time = xeeg[254 + batch_num]
            # print("plotting @%f seconds" % index_time)
            try:
                plt.ion()
                plt.figure(1)
                plt.clf()
                # plt.title("Plotting channel %d" % channel_num+1)

                plt.subplot(211)
                plt.grid()
                plt.xlabel("Frequency")
                plt.ylabel("Magnitude (db)")
                plt.title("Plotting channel %d at %f seconds" % (channel_num + 1, index_time))
                # plt.savefig("C:/testeeg/testeeg/mozart/logs/fft.png")
                #plt.semilogy(xf[0: batch_size / 2], T / batch_size * np.abs(current_batch_fft[0:batch_size / 2]) ** 2)
                plt.plot(xf[0: batch_size / 2], 20 * np.log(T / batch_size * np.abs(current_batch_fft[0:batch_size / 2]) ** 2))
                plt.ylim([-200, 300])

                plt.subplot(212)
                plt.grid()
                plt.xlabel("Time (s)")
                plt.ylabel("Magnitude (V)")
                plt.title("Plotting channel %d at %f seconds" % (channel_num + 1, index_time))
                plt.ylim([4050, 4500])
                # print(xeeg.__len__(), current_batch_eeg.__len__())

                plt.plot(xeeg_batch, current_batch_eeg)
                plt.pause(0.01)

                plt.show()

            except:
                print("Error streaming graphs")
                continue


def gettimeseriesdata(raw_data_all, batchsize, channel_bottom=5, channel_top=13, path_to_load=0):
    # If a path is provided to load pickled data, then attempt to load data from pickled files
    if path_to_load is not 0:
        print("\n-- Loading saved Time data --")
        fft_raw, spectro_raw, csd_raw, time_raw = load_pickled_freq_data(path_to_load, timeonly=True)
        if time_raw is not 0:
            np_time_raw = np.asarray(time_raw)
            print("Loaded time data: ", np_time_raw.shape)
            return np_time_raw
        else:
            print("Unable to load from pickled data at path %s" % path_to_load)
            option = input("Generate new raw data? y/n\n>>>>>> ")
            if option is 'y':
                print("Generating new data")
            else:
                return 0

    timestamp = str(time.time())
    rs_shape = np.shape(raw_data_all)
    if rs_shape[1] < 30:
        raw_data = raw_data_all
    else:
        raw_data = raw_data_all[:, channel_bottom:channel_top]

    raw_data = np.asarray(raw_data)
    raw_data_shape = np.shape(raw_data)

    eeg_ch_data = []

    for channel_num in range(0, raw_data_shape[1]):
        channel_data = raw_data[:, channel_num]
        eeg_data = []
        x_f = 0
        print("Channel %d starting" % (channel_num + 1))
        while x_f + batchsize < raw_data_shape[0]:
            # print("\tIndex: %d" % x_f)
            y = channel_data[x_f:x_f + batchsize]
            eeg_data.append(y)  # Append data from all batches in the current channel
            x_f += 1
        eeg_ch_data.append(eeg_data)

    np_eeg = np.asarray(eeg_ch_data)
    pickle_time_data(np_eeg, timestamp,
                     path_to_save="C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/X_eeg_fft/blink-ten/")
    return np_eeg


def getfft(raw_data_all, batchsize, channel_bottom=5, channel_top=13, path_to_load=0):
    # If a path is provided to load pickled data, then attempt to load data from pickled files
    if path_to_load is not 0:
        print("\n-- Loading saved FFT data --")
        fft_raw, spectro_raw, csd_raw, time_raw = load_pickled_freq_data(path_to_load)
        if fft_raw is not -1:
            np_fft_raw = np.asarray(fft_raw)
            print("Loaded fft data: ", np_fft_raw.shape)
            return 0, np_fft_raw, spectro_raw, csd_raw
        else:
            print("Unable to load from pickled data at path %s" % path_to_load)
            option = input("Generate new raw data? y/n\n>>>>>> ")
            if option is 'y':
                print("Generating new data")
            else:
                return 0

    timestamp = str(time.time())

    # Define the channels to get from the 14 channel raw data
    rs_shape = np.shape(raw_data_all)
    if rs_shape[1] < 30:
        raw_data = raw_data_all
    else:
        raw_data = raw_data_all[:, channel_bottom:channel_top]

    raw_data = np.asarray(raw_data)

    # Sampling frequency
    fs = 128.0

    # Time sampling interval in s
    T = 1 / fs

    # Number of samples
    n = rs_shape[0]

    # Get x values for time-based graph values
    xf = np.linspace(0.0, 1.0 / (2.0 * T), batchsize / 2)

    fft_channels = []

    increment_step = 1

    # Get fft of each EEG channel by batch size
    for i in range(0, rs_shape[1]):
        fft_data = []
        x_f = 0
        channel_data = raw_data[:, i]

        print("Channel %d starting" % (i + 1))
        while x_f + batchsize < n:
            # print("\tIndex: %d" % x_f)
            y = channel_data[x_f:x_f + batchsize]
            yf = fft(y)
            fft_data.append(yf)  # Append data from all batches in the current channel
            x_f += increment_step

        fft_channels.append(fft_data)  # Append data from all channels together

    np_fft = np.asarray(fft_channels)

    status = pickle_freq_data(fft_channels, 1, 1, timestamp,
                              path_to_save="C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/X_eeg_fft/blink-ten/")
    #        spectro_data = plot_spectrogram(raw_data, n, fs, channel_bottom + 1, print_frequency_graph)
    #        csd_data = plot_csd(raw_data, n, fs, channel_bottom + 1, print_frequency_graph)

    print(status)
    return xf, np_fft, 0, 0


def getfreqmag_batch(batched_freq_raw_data, period):
    batch_shape = np.shape(np.asarray(batched_freq_raw_data))
    sample_freq = 1.0 / period
    freq_incr = sample_freq / 2.0 / batch_shape[0]
    # delta = 0.5Hz - 3Hz
    # theta = 3Hz - 8Hz
    # alpha = 8Hz - 12Hz
    # beta = 12Hz - 38Hz
    # gamma = 38Hz - 42Hz

    xf = np.linspace(0, 64, 128)
    batch_power = 20 * np.log(period / batch_shape[0] * np.abs(batched_freq_raw_data[0:batch_shape[0] / 2]) ** 2)

    alpha_indices = [int(8 * batch_shape[0]/2 / (sample_freq / 2)), int(12 * batch_shape[0]/2 / (sample_freq / 2))]
    alpha_values = batch_power[alpha_indices[0]:alpha_indices[1]]
    alpha_values_avg = np.mean(alpha_values)

    delta_indices = [int(0.5 * batch_shape[0]/2 / (sample_freq / 2)), int(3 * batch_shape[0]/2 / (sample_freq / 2))]
    delta_values = batch_power[delta_indices[0]:delta_indices[1]]
    delta_values_avg = np.mean(delta_values)

    theta_indices = [int(3 * batch_shape[0]/2 / (sample_freq / 2)), int(8 * batch_shape[0]/2 / (sample_freq / 2))]
    theta_values = batch_power[theta_indices[0]:theta_indices[1]]
    theta_values_avg = np.mean(theta_values)

    gamma_indices = [int(38 * batch_shape[0]/2 / (sample_freq / 2)), int(42 * batch_shape[0]/2 / (sample_freq / 2))]
    gamma_values = batch_power[gamma_indices[0]:gamma_indices[1]]
    gamma_values_avg = np.mean(gamma_values)

    beta_indices = [int(12 * batch_shape[0]/2 / (sample_freq / 2)), int(38 * batch_shape[0]/2 / (sample_freq / 2))]
    beta_values = batch_power[beta_indices[0]:beta_indices[1]]
    beta_values_avg = np.mean(beta_values)







    print(delta_values_avg, theta_values_avg, alpha_values_avg, beta_values_avg, gamma_values_avg)








def plot_spectrogram(raw_data, nfft, fs, channel_bottom, print_frequency_graph):
    data_shape = raw_data.shape

    print("Generating spectrogram...")
    plt_num = 1
    plt.clf()
    plt.figure(1)

    channel_data = []
    for i in range(0, data_shape[1]):
        plt.subplot(8, 2, plt_num)

        f, t, Sxx = signal.spectrogram(x=raw_data[:, i], nfft=nfft, fs=fs, noverlap=127, nperseg=128,
                                       scaling='density')  # returns PSD power per Hz
        plt.pcolormesh(t, f, Sxx)

        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Channel %s' % (i + channel_bottom))
        plt_num += 1
        channel_data.append([f, t, Sxx])
        print("\tChannel %d spectrogram generated" % i)
    if print_frequency_graph:
        plt.show()
    return channel_data


def plot_csd(raw_data, nfft, fs, channel_bottom, print_frequency_graph):
    data_shape = raw_data.shape
    channel_data = []

    print("Generating cross spectral density graph...")
    plt_num = 1
    plt.clf()
    plt.figure(1)
    for i in range(0, data_shape[1] - 1):
        plt.subplot(8, 2, plt_num)
        x = raw_data[:, i]
        y = raw_data[:, i + 1]
        f, Pxy = signal.csd(x, y, nfft=nfft, fs=fs)  # returns PSD power per Hz
        plt.semilogy(f, np.abs(Pxy))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('CSD [V**2/Hz]')
        plt.title('Channel %s' % (i + channel_bottom))
        plt_num += 1
        channel_data.append([f, Pxy])

    if print_frequency_graph:
        plt.show()
    return channel_data


def getdatasets_blinkonce():
    eeg_input, output = face.get_blink_once_ds()
    return eeg_input[2], output


def getdatasets_blink_ten():
    eeg_input, output = face.get_blink_ten_ds()
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
