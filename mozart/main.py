import eeg_data.main
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils, serializers, Chain
import chainer.links as L
import chainer.functions as F
from scipy import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import pickle
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# noinspection PyMethodMayBeStatic,PyUnusedLocal,PyShadowingNames,PyTypeChecker
class eeg_learner:
    outputfile = ""

    def __init__(self, workingdirectory):
        self.workingdirectory = workingdirectory
        self.keyword = ""
        self.optimizer = optimizers.SGD(lr=0.01)
        self.model_path = self.workingdirectory + '/mozart/models/' + self.keyword + ".lr" + str(
                self.optimizer.lr) + ".model"
        self.model_to_use = FunctionSet(
                x_h1=F.Linear(448, 512),
                h1_h2=F.Linear(512, 1024),
                h2_h3=F.Linear(1024, 1024),
                h3_h4=F.Linear(1024, 512),
                h4_y=F.Linear(512, 32),
        )

    # noinspection PyMethodMayBeStatic
    def getmodel(self, batchsize=1):
        print("\n-- Loading new model --")

        eeg_model = FunctionSet(
                x_h1=F.Linear(14 * batchsize, 28 * batchsize),
                h1_h2=F.Linear(28 * batchsize, 32 * batchsize),
                h2_h3=F.Linear(32 * batchsize, 8 * batchsize),
                h3_h4=F.Linear(8 * batchsize, 4 * batchsize),
                h4_y=F.Linear(4 * batchsize, batchsize),
        )

        optimizer = optimizers.MomentumSGD(lr=0.01)
        optimizer.setup(eeg_model)
        print("Model Loaded!\n")
        return eeg_model, optimizer

    def set_components(self, model, optimizer):
        self.model_to_use = model
        self.optimizer = optimizer

    def getmodel_spectro(self, lm):
        eeg_model = L.Classifier(lm)
        return eeg_model

    def savernn(self, model=0):
        if model is 0:
            serializers.save_npz(self.model_path, self.model_to_use)
            pickle.dump(self.model_to_use, open(self.model_path, 'wb'))
        else:
            serializers.save_npz(self.model_path, model)
            pickle.dump(model, open(self.model_path, 'wb'))

    # noinspection PyBroadException
    def get_savedrnn(self, path_to_load=0):
        if path_to_load is 0:
            try:
                eeg_model = pickle.load(open(self.model_path, "rb"))
                # serializers.load_npz(self.model_path, eeg_model)
                return eeg_model
            except Exception as e:
                print("Error loading saved NN: ", e)
                return -1
        else:
            try:
                eeg_model = pickle.load(open(path_to_load, "rb"))
                # serializers.load_npz(self.model_path, eeg_model)
                return eeg_model
            except Exception as e:
                print("Error loading saved NN: ", e)
                return -1

    # noinspection PyCallingNonCallable
    def forward(self, x_batch_curr, y_batch_curr, w=0, volatile=False):


        current_sample = Variable(x_batch_curr, volatile=volatile)

        y_batch_curr = np.asarray(y_batch_curr).reshape(1, -1)
        current_output = Variable(y_batch_curr, volatile=volatile)

        h1_current = F.sigmoid(self.model_to_use.x_h1(current_sample))

        h2_current = F.sigmoid(self.model_to_use.h1_h2(h1_current))
        h2_weighted = np.dot(h2_current, w)

        h3_current = F.sigmoid(self.model_to_use.h2_h3(h2_weighted))

        h4_current = F.sigmoid(self.model_to_use.h3_h4(h3_current))
        h4 = h4_current
        y = self.model_to_use.h4_y(h4)

        loss = F.sigmoid_cross_entropy(y, current_output)
        current_output.data = np.squeeze(current_output.data)

        y.data = y.data.reshape(-1, 1)
        accuracy = F.accuracy(y, current_output)

        wdelta = np.asarray(w.data).reshape(-1, 1) * np.asarray(y.data).reshape(1, -1)
        w.data = wdelta.mean(1).reshape(1, -1)

        return accuracy, loss, y, w

    # noinspection PyCallingNonCallable
    def forward_eye_states(self, x_batch_curr, y_batch_curr, volatile):

        current_sample = Variable(x_batch_curr, volatile=volatile)

        y_batch_curr = np.asarray(y_batch_curr).reshape(32, -1)
        current_output = Variable(y_batch_curr, volatile=volatile)

        h1_current = F.sigmoid(self.model_to_use.x_h1(current_sample))

        h2_current = F.sigmoid(self.model_to_use.h1_h2(h1_current))

        h3_current = F.sigmoid(self.model_to_use.h2_h3(h2_current))

        h4_current = F.sigmoid(self.model_to_use.h3_h4(h3_current))

        h4 = h4_current
        y = self.model_to_use.h4_y(h4)

        y.data = y.data.reshape(32, -1)
        loss = F.sigmoid_cross_entropy(y, current_output)
        current_output.data = np.squeeze(current_output.data)

        accuracy = F.accuracy(y, current_output)

        return accuracy, loss, y

    def forward_spectrogram(self):
        pass

    def train_timeonly(self, savemodel=True):
        train_plot = plt.figure()
        plt1 = train_plot.add_subplot(211)
        plt2 = train_plot.add_subplot(212)

        # Get a fresh neural net and setup the optimizer with it
        self.model_to_use = self.getmodel()
        self.optimizer.setup(self.model_to_use)

        # Get EEG data to train with
        train_X_raw, train_Y = eeg_data.main.getdatasets_eeg()

        train_X = self.normalizevalues_eeg(train_X_raw)

        epochcount = 0
        datasize = train_X.shape[0]

        train_Y = self.gettargetdataset(target_state, datasize)

        epoch_sums = []
        epoch_loss = []
        epochs = []

        outputpath = "C:/Users/SourabhKatti/Documents/engine/mozart/logs" + self.outputfile + ".txt"
        sys.stdout = Logger(outputpath)
        #        for layer in self.model_to_use._children:
        #            print(layer[0], layer[1].W.shape)

        for epoch in range(20):
            epochcount += 1
            print('epoch %d' % epoch)
            sum_accuracy = 0.0
            sum_loss = 0.0
            batchsize = 32

            for i in range(2, datasize - batchsize):
                # Get current, previous and next sample from EEG recording to send through the net
                x_batch_curr = np.asarray(train_X[i:i + batchsize]).astype(np.float32).reshape(1, -1)

                # Get current, previous and next sample from target output ready
                y_batch_curr = np.asarray(train_Y[i:i + batchsize]).astype(np.int32)

                self.optimizer.zero_grads()
                accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += loss.data

                output.data = output.data.reshape(1, -1)

                loss.backward()
                self.optimizer.update()

            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

            # self.optimizer.lr += 0.00005

        plt1.plot(epochs, epoch_sums)
        plt2.plot(epochs, epoch_loss)
        pltimage = outputpath + '.png'
        train_plot.show()
        train_plot.savefig(pltimage)
        if savemodel:
            self.outputfile += str(np.max(epochs))
            self.savernn()
        return train_X

    def train_eyes_open_close(self):

        # Get a fresh neural net and setup the optimizer with it
        self.model_to_use = self.getmodel()
        self.optimizer.setup(self.model_to_use)

        epochs_total = 1

        self.train_eyes_open_only(epochs_total)
        self.train_eyes_closed_only(epochs_total)

    def train_eyes_open_only(self, epochs_total, savemodel=True):
        train_plot = plt.figure()
        plt1 = train_plot.add_subplot(211)
        plt2 = train_plot.add_subplot(212)

        target_state = 2

        # Get EEG data to train with
        train_X_raw = eeg_data.main.getdatasets_eyes_open()

        train_X = np.asarray(train_X_raw)
        train_shape = train_X.shape

        epochcount = 0
        datasize = train_shape[0]

        train_Y = self.gettargetdataset(target_state, datasize)

        epoch_sums = []
        epoch_loss = []
        epochs = []

        outputpath = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/eyes_open/" + self.outputfile + ".txt"
        sys.stdout = Logger(outputpath)
        print("\nStarting to train model on eyes_open samples.\n\tDataset size: ", train_shape)
        print("\tTarget state: ", target_state)

        for epoch in range(epochs_total):
            epochcount += 1
            print('epoch %d' % (epoch + 1))
            sum_accuracy = 0.0
            sum_loss = 0.0
            batchsize = 32

            for i in range(0, datasize - batchsize):
                # Get current, previous and next sample from EEG recording to send through the net
                x_batch_curr = np.asarray(train_X[i:i + batchsize]).astype(np.float32).reshape(1, -1)
                max_value = np.max(x_batch_curr)
                x_batch_curr = np.divide(x_batch_curr, max_value)

                # Get current, previous and next sample from target output ready
                y_batch_curr = np.asarray(train_Y[i:i + batchsize]).astype(np.int32)

                self.optimizer.zero_grads()
                accuracy, loss, output = self.forward_eye_states(x_batch_curr, y_batch_curr, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += loss.data

                output.data = output.data.reshape(1, -1)

                loss.backward()
                self.optimizer.update()

            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

        # self.optimizer.lr += 0.00005

        plt1.plot(epochs, epoch_sums)
        plt2.plot(epochs, epoch_loss)
        pltimage = outputpath + '.png'
        train_plot.show()
        train_plot.savefig(pltimage)
        if savemodel:
            self.outputfile += str(np.max(epochs))
            self.savernn()

    def train_eyes_closed_only(self, epochs_total, savemodel=True):
        train_plot = plt.figure()
        plt1 = train_plot.add_subplot(211)
        plt2 = train_plot.add_subplot(212)

        # Get EEG data to train with
        train_X_raw = eeg_data.main.getdatasets_eyes_closed()
        train_X = np.asarray(train_X_raw)
        train_shape = train_X.shape

        target_state = 1

        epochcount = 0
        datasize = train_shape[0]

        train_Y = self.gettargetdataset(target_state, datasize)

        epoch_sums = []
        epoch_loss = []
        epochs = []

        outputpath = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/eyes_close/" + self.outputfile + ".txt"
        sys.stdout = Logger(outputpath)

        print("\nStarting to train model on eyes_closed samples.\n\tDataset size: ", train_shape)
        print("\tTarget state: ", target_state)

        for epoch in range(epochs_total):
            epochcount += 1
            print('epoch %d' % (epoch + 1))
            sum_accuracy = 0.0
            sum_loss = 0.0
            batchsize = 32

            for i in range(0, datasize - batchsize):
                # Get current, previous and next sample from EEG recording to send through the net
                x_batch_curr = np.asarray(train_X[i:i + batchsize]).astype(np.float32).reshape(1, -1)
                max_value = np.max(x_batch_curr)
                x_batch_curr = np.divide(x_batch_curr, max_value)

                # Get current, previous and next sample from target output ready
                y_batch_curr = np.asarray(train_Y[i:i + batchsize]).astype(np.int32)

                self.optimizer.zero_grads()
                accuracy, loss, output = self.forward_eye_states(x_batch_curr, y_batch_curr, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += loss.data

                output.data = output.data.reshape(1, -1)

                loss.backward()
                self.optimizer.update()

            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

        # self.optimizer.lr += 0.00005

        plt1.plot(epochs, epoch_sums)
        plt2.plot(epochs, epoch_loss)
        pltimage = outputpath + '.png'
        train_plot.show()
        train_plot.savefig(pltimage)
        if savemodel:
            self.outputfile += str(np.max(epochs))
            self.savernn()

    # noinspection PyShadowingNames
    def gettargetdataset(self, target_state, size):
        return np.full((size, 1), target_state, dtype=int)

    def train_timefreq(self, print_frequency_graph=True):

        # spectro_data = [f, t, Sxx]

        # Specify the range of channels to grab from raw EEG input
        channel_lower = 5
        channel_upper = 13

        # Get raw data
        train_X_raw = eeg_data.main.getdatasets_blink_ten()

        # Get fft values
        xf, fft_data, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw, print_frequency_graph,
                                                                    channel_bottom=channel_lower)

        fft_data = np.asarray(fft_data)
        xf = np.asarray(xf)

        # Print details about the FFT, spectrogram and CSD datasets
        self.print_fft_data(xf, fft_data, channel_lower)
        spectro_shape = self.print_spectro_data(spectro_data, channel_lower)

    def train_time_batched(self, num_epochs, batchsize, x, y, savemodel=True, print_frequency_graph=True):

        self.keyword = "time-" + str(num_epochs)


        # Get a fresh neural net and setup the optimizer with it
        eeg_model, eeg_optimizer = self.getmodel(batchsize)
        self.set_components(eeg_model, eeg_optimizer)

        train_X = x
        train_Y = y

        epochcount = 0
        datasize = x.shape[1]

        epoch_sums = []
        epoch_loss = []
        epochs = []

        x_datasize = np.linspace(0, datasize - 1, datasize)

        plot = False

        print("== Starting time training ==")

        for epoch in range(num_epochs):
            epochcount += 1
            print('epoch %d' % epoch)
            outputpath = self.workingdirectory + "/mozart/logs/blink-ten-" + str(epoch) + "-" + self.outputfile + ".txt"

            sum_accuracy = 0.0
            sum_loss = 0.0
            total_loss = []
            total_accuracy = []
            total_x = []

            batchnum = 0

            random.shuffle(x_datasize)


            sys.stdout = Logger(outputpath)
            w = Variable(np.random.rand(1, 32 * batchsize).astype(np.float32))

            for i in x_datasize:

                batchnum += 1

                # Get current, previous and next sample from EEG recording to send through the net
                x_batch_curr = np.asarray(train_X[:, i, :]).astype(np.float32).reshape(1, -1)

                # Get current, previous and next sample from target output ready
                y_batch_curr = np.asarray(train_Y[i: i + batchsize]).astype(np.int32)

                self.optimizer.zero_grads()
                accuracy, loss, output, w = self.forward(x_batch_curr, y_batch_curr, w, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += loss.data

                total_x.append(batchnum)
                total_loss.append(loss.data)
                total_accuracy.append(accuracy.data)

                # print(sum_loss / i, sum_accuracy / i)
                if plot:

                    plt.figure(1)
                    plt.title("Training epoch %d" % (epoch + 1))

                    plt.clf()
                    plt.ion()

                    plt.subplot(211)
                    plt.ylim([0.0, 1.2])
                    plt.plot(total_x, total_loss)
                    plt.xlabel("Sample number")
                    plt.ylabel("Loss (%)")
                    plt.title("Training epoch %d\nLoss: %f" % (epoch + 1, sum_loss / batchnum))

                    plt.subplot(212)
                    plt.grid()
                    plt.ylim([0.0, 1.2])
                    plt.plot(total_x, total_accuracy)
                    plt.xlabel("Sample number")
                    plt.ylabel("Accuracy (%)")
                    plt.title("Accuracy: %f" % (sum_accuracy / batchnum))

                    plt.pause(0.001)
                    plt.show()

                output.data = output.data.reshape(1, -1)

                loss.backward()
                w.backward()
                loss.unchain_backward()
                self.optimizer.update()

            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)

            pltimage = outputpath + '.png'
            plt.savefig(pltimage)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

            self.optimizer.lr += 0.0005

        plt.clf()
        plt.subplot(211)
        plt.plot(epochs, epoch_sums)
        plt.subplot(212)
        plt.plot(epochs, epoch_loss)
        pltimage = outputpath + "-time" + '.png'
        # train_plot.show()
        plt.savefig(pltimage)
        if savemodel:
            self.savernn()
        return self.model_to_use, self.optimizer

    def train_freq_batched(self, num_epochs, batchsize, x, y, print_frequency_graph=True, savemodel=True):
        # Sample frequency of raw data
        fs = 128.0
        T = 1.0 / fs

        self.keyword = "psd-" + str(num_epochs)
        epochcount = 0
        train_X = x
        train_Y = y

        epochs = []
        epoch_sums = []
        epoch_loss = []
        datasize = x.shape[1]
        fft_model, fft_optimizer = self.getmodel(batchsize)
        self.set_components(fft_model, fft_optimizer)

        # Generate a set of x values to graph fft.
        #   Number of x points is half of the batchsize
        #   Values of x points range from 0.0 to half of the sampling frequency
        xf = np.linspace(0.0, 1.0 / (2.0 * T), batchsize / 2)
        x_datasize = np.linspace(0, datasize - 1, datasize)
        w = Variable(np.random.rand(1, 32 * batchsize).astype(np.float32))

        print("\n== Starting PSD training ==")
        for epoch in range(num_epochs):
            epochcount += 1
            print('epoch %d' % epoch)
            sum_accuracy = 0.0
            sum_loss = 0.0
            total_loss = []
            total_accuracy = []
            total_x = []

            batchnum = 0


            random.shuffle(x_datasize)

            outputpath = self.workingdirectory + "/mozart/logs/blink-ten-psd" + str(
                    epoch) + "-" + self.outputfile + ".txt"
            sys.stdout = Logger(outputpath)

            plot = False

            for i in x_datasize:

                batchnum += 1

                # Get current sample from raw data to send through the net
                x_batch_curr = np.asarray(train_X[:, i, :]).astype(np.float32).reshape(1, -1)

                # Get current output from target output to send through the net
                y_batch_curr = np.asarray(train_Y[i: i + batchsize]).astype(np.int32)

                self.optimizer.zero_grads()
                # print(x_batch_curr)
                accuracy, loss, output, w = self.forward(x_batch_curr, y_batch_curr, w=w, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += loss.data

                total_x.append(batchnum)
                total_loss.append(loss.data)
                total_accuracy.append(accuracy.data)

                fft_graph = np.asarray(train_X[:, i, :]).astype(np.float32)

                # print(sum_loss / i, sum_accuracy / i)
                if plot:
                    plt.title("Training epoch %d" % (epoch + 1))
                    plt.figure(1)

                    plt.clf()
                    plt.ion()

                    np_x = np.asarray(fft_graph)
                    plt.subplot(311)
                    plt.ylim([-15, 25])
                    plt.grid()
                    plt.plot(xf, np_x[0][0:batchsize / 2])
                    plt.ylabel("Power density (db/Hz)(%)")
                    plt.title("Frequency (Hz)")

                    plt.subplot(312)
                    plt.grid()
                    plt.ylim([0.0, 1.2])
                    plt.plot(total_x, total_loss)
                    plt.ylabel("Loss (%)")
                    plt.title("Training epoch %d\nLoss: %f" % (epoch + 1, sum_loss / batchnum))

                    plt.subplot(313)
                    plt.grid()
                    plt.ylim([0.0, 1.2])
                    plt.plot(total_x, total_accuracy)
                    plt.xlabel("Sample number")
                    plt.ylabel("Accuracy (%)")
                    plt.title("Accuracy: %f" % (sum_accuracy / batchnum))

                    plt.pause(0.001)
                    plt.show()

                output.data = output.data.reshape(1, -1)
                print(output.data.shape)

                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)

            pltimage = outputpath + '-freq.png'
            # train_plot.show()
            plt.savefig(pltimage)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

            self.optimizer.lr += 0.0005

        plt.clf()
        plt.subplot(211)
        plt.plot(epochs, epoch_sums)
        plt.subplot(212)
        plt.plot(epochs, epoch_loss)
        pltimage = "fft-" + outputpath + '.png'
        # train_plot.show()
        plt.savefig(pltimage)

        if savemodel:
            self.savernn(fft_model)
        return fft_model, fft_optimizer

    def train_blink_ten(self, batchsize, print_frequency_graph=True, savemodel=False):
        print("Training on BLINK-TEN dataset\n")

        # Get raw data
        train_X_raw, train_Y = eeg_data.main.getdatasets_blink_ten()
        for index, value in enumerate(train_Y):
            print(index, value)

        path_fft_output = self.workingdirectory + '/eeg_data/face/edf/X_eeg_fft/blink-ten/'

        # Get fft values
        # Load fft + time data
        xf, fft_data, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw, batchsize,
                                                                    path_to_load=path_fft_output)

        # Generate new set of fft + time data
        # xf, fft_data, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw, 256)
        batched_eeg_data = eeg_data.main.gettimeseriesdata(train_X_raw, batchsize, path_to_load=path_fft_output)

        psd_fft = eeg_data.main.get_fft_psd(fft_data, batchsize)

        print("\nFinal output-fft batched shape: ", fft_data.shape)
        print("Final output-time batched shape: ", batched_eeg_data.shape)

        # eeg_data.main.streamfft(fft_data, batched_eeg_data, batchsize)

        fft_data = np.asarray(fft_data)
        xf = np.asarray(xf)

        print("\n** -- Starting BLINK-TEN training -- **")

        # Number of epochs to run
        num_epochs = 1

        eeg_model, eeg_optimizer = self.train_time_batched(num_epochs, batchsize, batched_eeg_data, train_Y, savemodel=savemodel,
                                                           print_frequency_graph=print_frequency_graph)

        # eeg_model, eeg_optimizer = self.train_time_batched(batchsize, batched_eeg_data, train_Y, savemodel=savemodel, print_frequency_graph=print_frequency_graph )

        # self.keyword = "time-1"
        #
        # # Get a fresh neural net and setup the optimizer with it
        # eeg_model, eeg_optimizer = self.getmodel(batchsize)
        # self.set_components(eeg_model, eeg_optimizer)
        #
        # train_X = batched_eeg_data
        #
        # epochcount = 0
        # datasize = batched_eeg_data.shape[1]
        #
        # epoch_sums = []
        # epoch_loss = []
        # epochs = []
        #
        # plot = False
        #
        # print("== Starting time training ==")
        #
        # for epoch in range(40):
        #     epochcount += 1
        #     print('epoch %d' % epoch)
        #     sum_accuracy = 0.0
        #     sum_loss = 0.0
        #     total_loss = []
        #     total_accuracy = []
        #     total_x = []
        #
        #     batchnum = 0
        #
        #     x_datasize = np.linspace(0, datasize - 1, datasize)
        #     random.shuffle(x_datasize)
        #
        #     outputpath = self.workingdirectory + "/mozart/logs/blink-ten-" + str(epoch) + "-" + self.outputfile + ".txt"
        #     sys.stdout = Logger(outputpath)
        #
        #     for i in x_datasize:
        #
        #         batchnum += 1
        #
        #         # Get current, previous and next sample from EEG recording to send through the net
        #         x_batch_curr = np.asarray(train_X[:, i, :]).astype(np.float32).reshape(1, -1)
        #
        #         # Get current, previous and next sample from target output ready
        #         y_batch_curr = np.asarray(train_Y_raw[i: i + batchsize]).astype(np.int32)
        #
        #         self.optimizer.zero_grads()
        #         accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=False)
        #
        #         sum_accuracy += accuracy.data
        #         sum_loss += loss.data
        #
        #         total_x.append(batchnum)
        #         total_loss.append(loss.data)
        #         total_accuracy.append(accuracy.data)
        #
        #         # print(sum_loss / i, sum_accuracy / i)
        #         if plot:
        #             plt.title("Training epoch %d" % (epoch + 1))
        #             plt.figure(1)
        #
        #             plt.clf()
        #             plt.ion()
        #
        #             plt.subplot(211)
        #             plt.ylim([0.0, 1.2])
        #             plt.plot(total_x, total_loss)
        #             plt.xlabel("Sample number")
        #             plt.ylabel("Loss (%)")
        #             plt.title("Training epoch %d\nLoss: %f" % (epoch + 1, sum_loss / batchnum))
        #
        #             plt.subplot(212)
        #             plt.grid()
        #             plt.ylim([0.0, 1.2])
        #             plt.plot(total_x, total_accuracy)
        #             plt.xlabel("Sample number")
        #             plt.ylabel("Accuracy (%)")
        #             plt.title("Accuracy: %f" % (sum_accuracy / batchnum))
        #
        #             plt.pause(0.001)
        #             plt.show()
        #
        #         output.data = output.data.reshape(1, -1)
        #
        #         loss.backward()
        #         loss.unchain_backward()
        #         self.optimizer.update()
        #
        #     print("\tAccuracy of training: ", sum_accuracy / datasize)
        #     print("\tLoss of training: ", sum_loss / datasize)
        #
        #     pltimage = outputpath + '.png'
        #     # train_plot.show()
        #     plt.savefig(pltimage)
        #
        #     epoch_sums.append(sum_accuracy / datasize)
        #     epoch_loss.append(sum_loss / datasize)
        #     epochs.append(epoch)
        #
        #     self.optimizer.lr += 0.0005
        #
        # plt.clf()
        # plt.subplot(211)
        # plt.plot(epochs, epoch_sums)
        # plt.subplot(212)
        # plt.plot(epochs, epoch_loss)
        # pltimage = outputpath + '.png'
        # # train_plot.show()
        # train_plot.savefig(pltimage)
        # if savemodel:
        #     self.outputfile += str(np.max(epochs))
        #     self.savernn()

        fft_model, fft_optimizer = self.train_freq_batched(num_epochs, batchsize, psd_fft, train_Y, savemodel=savemodel,
                                                           print_frequency_graph=print_frequency_graph)

        # print("\n== Starting PSD training ==")
        # self.keyword = "psd-1"
        # epochcount = 0
        # train_X = psd_fft
        # epochs = []
        # epoch_sums = []
        # epoch_loss = []
        # fft_model, fft_optimizer = self.getmodel(batchsize)
        # self.set_components(fft_model, fft_optimizer)
        #
        # xf = np.linspace(0.0, 1.0 / (2.0 * T), batchsize / 2)
        #
        # for epoch in range(40):
        #     epochcount += 1
        #     print('epoch %d' % epoch)
        #     sum_accuracy = 0.0
        #     sum_loss = 0.0
        #     total_loss = []
        #     total_accuracy = []
        #     total_x = []
        #
        #     batchnum = 0
        #
        #     x_datasize = np.linspace(0, datasize - 1, datasize)
        #     random.shuffle(x_datasize)
        #
        #     outputpath = self.workingdirectory + "/mozart/logs/blink-ten-psd" + str(
        #             epoch) + "-" + self.outputfile + ".txt"
        #     sys.stdout = Logger(outputpath)
        #
        #     for i in x_datasize:
        #
        #         batchnum += 1
        #
        #         # Get current, previous and next sample from EEG recording to send through the net
        #         x_batch_curr = np.asarray(train_X[:, i, :]).astype(np.float32).reshape(1, -1)
        #
        #         # Get current, previous and next sample from target output ready
        #         y_batch_curr = np.asarray(train_Y_raw[i: i + batchsize]).astype(np.int32)
        #
        #         self.optimizer.zero_grads()
        #         # print(x_batch_curr)
        #         accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=False)
        #
        #         sum_accuracy += accuracy.data
        #         sum_loss += loss.data
        #
        #         total_x.append(batchnum)
        #         total_loss.append(loss.data)
        #         total_accuracy.append(accuracy.data)
        #
        #         fft_graph = np.asarray(train_X[:, i, :]).astype(np.float32)
        #
        #         # print(sum_loss / i, sum_accuracy / i)
        #         if plot:
        #             plt.title("Training epoch %d" % (epoch + 1))
        #             plt.figure(1)
        #
        #             plt.clf()
        #             plt.ion()
        #
        #             np_x = np.asarray(fft_graph)
        #             plt.subplot(311)
        #             plt.ylim([-15, 25])
        #             plt.grid()
        #             plt.plot(xf, np_x[0][0:batchsize / 2])
        #             plt.ylabel("Power density (db/Hz)(%)")
        #             plt.title("Frequency (Hz)")
        #
        #             plt.subplot(312)
        #             plt.grid()
        #             plt.ylim([0.0, 1.2])
        #             plt.plot(total_x, total_loss)
        #             plt.ylabel("Loss (%)")
        #             plt.title("Training epoch %d\nLoss: %f" % (epoch + 1, sum_loss / batchnum))
        #
        #             plt.subplot(313)
        #             plt.grid()
        #             plt.ylim([0.0, 1.2])
        #             plt.plot(total_x, total_accuracy)
        #             plt.xlabel("Sample number")
        #             plt.ylabel("Accuracy (%)")
        #             plt.title("Accuracy: %f" % (sum_accuracy / batchnum))
        #
        #             plt.pause(0.001)
        #             plt.show()
        #
        #         output.data = output.data.reshape(1, -1)
        #
        #         loss.backward()
        #         self.optimizer.update()
        #
        #     print("\tAccuracy of training: ", sum_accuracy / datasize)
        #     print("\tLoss of training: ", sum_loss / datasize)
        #
        #     pltimage = outputpath + '.png'
        #     # train_plot.show()
        #     plt.savefig(pltimage)
        #
        #     epoch_sums.append(sum_accuracy / datasize)
        #     epoch_loss.append(sum_loss / datasize)
        #     epochs.append(epoch)
        #
        #     self.optimizer.lr += 0.0005
        #
        # plt.clf()
        # plt.subplot(211)
        # plt.plot(epochs, epoch_sums)
        # plt.subplot(212)
        # plt.plot(epochs, epoch_loss)
        # pltimage = outputpath + '.png'
        # # train_plot.show()
        # train_plot.savefig(pltimage)
        # self.set_components(eeg_model, eeg_optimizer)
        # if savemodel:
        #     self.outputfile = self.outputfile + str(np.max(epochs)) + "-fft"
        #     self.savernn(fft_model)
        #
        #     # Print details about the FFT, spectrogram and CSD datasets
        #     # self.print_fft_data(xf, fft_data, channel_lower)
        #     # spectro_shape = self.print_spectro_data(spectro_data, channel_lower)
        self.set_components(eeg_model, eeg_optimizer)

    def print_fft_data(self, xf, fft_data, channel_lower):
        i = channel_lower
        print("\nFFT data")
        print("Time-series points: ", xf.__len__())
        for channel in fft_data:
            i += 1
            print("Channel %d FFT size: %d" % (i, channel.__len__()))

    def print_spectro_data(self, spectro_data, channel_lower):
        i = channel_lower
        print("\nSpectrogram output")
        freqshape = []
        for channel in spectro_data:
            i += 1
            freqs, t, sxx = channel
            freqs = np.asarray(freqs)
            t = np.asarray(t)
            sxx = np.asarray(sxx)
            print("Channel %d data" % i)
            print("\tFrequency: ", freqs.shape)
            print("\t\tMin: ", np.min(freqs))
            print("\t\tMax: ", np.max(freqs))
            print("\tTime: ", t.shape)
            print("\t\tMin: ", np.min(t))
            print("\t\tMax: ", np.max(t))
            print("\tSpectrogram: ", sxx.shape)
            print("\t\tMin: ", np.min(sxx))
            print("\t\tMax: ", np.max(sxx))
            freqshape = freqs.shape
        return freqshape

    def normalizevalues_eeg(self, raw_eeg):

        x, y = raw_eeg.shape
        eeg_norm = np.zeros(shape=[x, y])

        channel_num = 0

        while channel_num < y:
            raw_sample = raw_eeg[:, channel_num]
            mean_value = np.mean(raw_sample)
            norm_sample = []
            for sample_index in range(x):
                value = raw_sample[sample_index]
                norm_sample = float(value) - float(mean_value)
                eeg_norm[sample_index, channel_num] = norm_sample
            channel_num += 1

        return eeg_norm

    def test_models(self, batchsize):
        # test_X, test_Y = eeg_data.main.getdatasets_eeg()
        print("\n-- Starting testing --")

        path_fft_output = os.path.realpath(self.workingdirectory) + "/eeg_data/face/edf/X_eeg_fft/blink-ten/"

        train_X_raw, train_Y_raw = eeg_data.main.getdatasets_blink_ten()
        xf, fft_data, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw, batchsize,
                                                                    path_to_load=path_fft_output)

        # Generate new set of fft + time data
        # xf, fft_data, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw, 256)
        batched_eeg_data = eeg_data.main.gettimeseriesdata(train_X_raw, batchsize, path_to_load=path_fft_output)

        test_X = batched_eeg_data
        test_Y = train_Y_raw
        datasize = test_X.shape[1]

        sum_accuracy = 0.0
        sum_loss = 0.0

        sample_preds = []

        # Check if there is a saved model to use and load it if there is
        if self.get_savedrnn() != -1:
            self.model_to_use = self.get_savedrnn()
            self.optimizer.setup(self.model_to_use)
            print("Model loaded!")

        for i in range(0, datasize):
            x_batch_curr = np.asarray(test_X[:, i, :]).astype(np.float32).reshape(1, -1)

            # Get current, previous and next sample from target output ready
            y_batch_curr = np.asarray(test_Y[i:i + batchsize]).astype(np.int32)

            self.optimizer.zero_grads()
            accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=True)

            sum_accuracy += accuracy.data
            sum_loss += loss.data

            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

            print(x_batch_curr.shape, x_batch_curr[0], y_batch_curr[0])

            sample_preds.append(output.data)

        min_preds = np.min(sample_preds)
        normalized_preds = []
        for value in sample_preds:
            normalized_preds.append(np.divide(value, min_preds).astype(float))
        # max_preds = np.max(normalized_preds)
        # normalized_max_preds = np.divide(normalized_preds, max_preds)

        return test_X, normalized_preds, test_Y

    def test_eye_states_model(self):
        test_X_raw = eeg_data.main.getdatasets_test_eye_states()
        test_X = np.asarray(test_X_raw)

        datasize = test_X.shape[0]

        sum_accuracy = 0.0
        sum_loss = 0.0

        sample_preds = []
        batchsize = 32

        # Check if there is a saved model to use and load it if there is
        if self.get_savedrnn() != -1:
            self.model_to_use = self.get_savedrnn()
            self.optimizer.setup(self.model_to_use)

        for i in range(2, datasize - batchsize):
            x_batch_curr = np.asarray(test_X[i:i + batchsize]).astype(np.float32).reshape(1, -1)
            y_batch_curr = np.zeros_like(x_batch_curr).astype(np.int32)
            max_value = np.max(x_batch_curr)
            x_batch_curr = np.divide(x_batch_curr, max_value)

            self.optimizer.zero_grads()

            accuracy, loss, output = self.forward_eye_states(x_batch_curr, y_batch_curr, volatile=True)

            sum_accuracy += accuracy.data
            sum_loss += loss.data

            loss.backward()
            self.optimizer.update()

            sample_preds.append(output.data)

        min_preds = np.min(sample_preds)
        normalized_preds = []
        for value in sample_preds:
            normalized_preds.append(np.divide(value, min_preds).astype(float))
        # max_preds = np.max(normalized_preds)
        # normalized_max_preds = np.divide(normalized_preds, max_preds)

        return test_X, normalized_preds

    def evaluate(self, dataset):
        # Evaluation routine
        evaluator = self.model_freq.copy()  # to use different state
        evaluator.predictor.reset_state()  # initialize state
        evaluator.predictor.train = False  # dropout does nothing

        sum_log_perp = 0
        for i in range(dataset.size - 1):
            x = Variable(np.asarray(np.abs(dataset[i:i + 1])).astype(np.int), volatile='on')
            t = Variable(np.asarray(np.abs(dataset[i + 1:i + 2])).astype(np.int), volatile='on')
            loss = evaluator(x, t)
            sum_log_perp += loss.data
        return math.exp(float(sum_log_perp) / (dataset.size - 1))

    # Refine the predictions and plot the results along with the original EEG waveforms
    def plot_predictions(self, raw_data, predictions, targetoutputs, batchsize):
        np_preds = np.asarray(predictions).reshape(-1, batchsize)
        np_preds_shape = np_preds.shape
        print("Predictions shape: ", np_preds_shape)
        # Setup a blank array of 0s based on the number of samples + batchsize

        preds = np.zeros(np.max(np_preds.shape) + batchsize)
        indices = []
        raw_data_indices = []
        raw_data_values = []

        # Output preds have size [1, batchsize] so get the average of all the values
        # for i in range(0, np_preds_shape[0]):
        #     for p in range(0, np_preds_shape[1]):
        #         preds[i + p] += predictions[i, p, :]
        #         # print(i, preds)
        # preds = np.divide(preds, batchsize)

        # Average the output prediction signal over a specified window
        pred_size = preds.__len__()

        # avg_preds = []
        # i = 0
        # averaging_window = 1
        # print("Averaging predictions with a %d sample window" % averaging_window)
        # while i < pred_size - averaging_window:
        #     avg_pred = np.mean(preds[i:i + averaging_window])
        #     avg_preds.append(avg_pred)
        #     i += 1
        # preds[0:pred_size - averaging_window] = avg_preds
        # min_pred = np.min(avg_preds)
        # max_pred = np.max(avg_preds)
        # preds_norm = np.divide(preds, max_pred)
        # preds_norm *= 2

        # Get raw EEG data to graph
        # for e, value in enumerate(raw_data):
        #     raw_data_indices.append(e)
        #     raw_data_values.append(value)

        # predictions = np.asarray(preds_norm).reshape(-1, 1)
        # raw_data_values = np.asarray(raw_data_values).reshape(14, -1, 256)
        # raw_data_indices = np.asarray(raw_data_indices).reshape(-1, 1)
        # raw_data_shape = raw_data_values.shape

        # Plot the predictions and raw EEG signals
        predsfile = self.workingdirectory + "/mozart/logs/" + self.outputfile + "-preds.png"
        preds_to_graph = []

        for i in range(0, np_preds_shape[0]):
            indices.append(i)
            preds_to_graph.append(np.max(preds[i:i + batchsize]))
            plt.figure(1)

            plt.clf()
            plt.ion()

            plt.plot(indices, preds_to_graph)
            plt.xlabel("Sample number")
            plt.ylabel("Prediction (%)")
            plt.title("Prediction Results")

            plt.pause(0.001)
            plt.show()

        data_index = 0
        plt.savefig(predsfile)


        # Setup an RNN using chainer library

# eeg_learner = eeg_learner()
#
# # training_data = eeg_learner.train_timeonly()
# # raw_data, predictions = eeg_learner.test_eeg_sample()
# # eeg_learner.plot_predictions(raw_data, predictions)
#
# # eeg_learner.train_eyes_open_close()
# # test_input, predictions = eeg_learner.test_eye_states_model()
# # eeg_learner.plot_predictions(test_input, predictions)
#
#
#
# batchsize = 256
# # eeg_learner.train_blink_ten(batchsize, print_frequency_graph=True, savemodel=True)
# raw_data, predictions, targetoutputs = eeg_learner.test_eeg_sample(batchsize)
# eeg_learner.plot_predictions(raw_data, predictions, targetoutputs, batchsize)
