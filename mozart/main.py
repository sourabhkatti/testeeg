import eeg_data.main
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils, Chain
import chainer.functions as F
import chainer.links as L
from scipy import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# noinspection PyPep8Naming
class eeg_learner():
    outputfile = "my"
    model_to_use = FunctionSet(
            x_h1=F.Linear(448, 512),
            h1_h2=F.Linear(512, 1024),
            h2_h3=F.Linear(1024, 1024),
            h3_h4=F.Linear(1024, 512),
            h4_y=F.Linear(512, 32),
    )
    optimizer = optimizers.SGD(lr=0.05)
    model_path = "C:/Users/SourabhKatti/Documents/engine/mozart/models/" + outputfile + ".model"

    def __init__(self):
        self.initialsetup()

    def initialsetup(self):
        self.outputfile = str(time.time())

    def getmodel_eeg(self):
        eeg_model = FunctionSet(
                x_h1=F.Linear(448, 512),
                h1_h2=F.Linear(512, 1024),
                h2_h3=F.Linear(1024, 1024),
                h3_h4=F.Linear(1024, 512),
                h4_y=F.Linear(512, 32),
        )
        return eeg_model

    def getmodel_lstm(self, n_units):
        eeg_model = FunctionSet(x_h1=L.Linear(n_units, 4 * n_units),
                                h1_h2=L.Linear(4 * n_units, 4 * n_units),
                                h2_h3=L.Linear(4 * n_units, 4 * n_units),
                                h3_h4=L.Linear(4 * n_units, 4 * n_units),
                                h4_y=L.Linear(4 * n_units, 4 * n_units))
        return eeg_model

    def forward_lstm(self, c, train_X, prev_Y, volatile=False):

        current_sample = Variable(train_X, volatile=volatile)

        y_batch_curr = np.asarray(prev_Y).reshape(1, -1)
        current_output = Variable(y_batch_curr, volatile=volatile)

        h1_current = F.sigmoid(self.model_to_use.x_h1(current_sample))

        h2_current = F.sigmoid(self.model_to_use.h1_h2(h1_current))

        h3_current = F.sigmoid(self.model_to_use.h2_h3(h2_current))

        h4_current = F.sigmoid(self.model_to_use.h3_h4(h3_current))

        h4 = self.model_to_use.h4_y(h4_current)

        c, y = F.lstm(c, h4)

        c.data = np.asarray(c.data).astype(int32)

        loss = F.sigmoid_cross_entropy(y, c)
        y.data = y.data.reshape(-1, 1)
        accuracy = F.accuracy(y, current_output)

        return accuracy, loss, c, y

    def forward(self, x_batch_curr, y_batch_curr, volatile=False):
        accuracy = 0
        loss = 0

        current_sample = Variable(x_batch_curr, volatile=volatile)

        y_batch_curr = np.asarray(y_batch_curr).reshape(1, -1)
        current_output = Variable(y_batch_curr, volatile=volatile)

        h1_current = F.sigmoid(self.model_to_use.x_h1(current_sample))

        h2_current = F.sigmoid(self.model_to_use.h1_h2(h1_current))

        h3_current = F.sigmoid(self.model_to_use.h2_h3(h2_current))

        h4_current = F.sigmoid(self.model_to_use.h3_h4(h3_current))

        h4 = h4_current
        y = self.model_to_use.h4_y(h4)

        loss = F.sigmoid_cross_entropy(y, current_output, normalize=True)
        current_output.data = np.squeeze(current_output.data)
        y.data = y.data.reshape(-1, 1)
        accuracy = F.accuracy(y, current_output)

        return accuracy, loss, y

    def forward_eye_states(self, x_batch_curr, y_batch_curr, volatile):
        accuracy = 0
        loss = 0

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

    def train_timeonly(self, lstm=False, savemodel=True):
        train_plot = plt.figure()
        plt1 = train_plot.add_subplot(211)
        plt2 = train_plot.add_subplot(212)

        batchsize = 32

        ## Get a fresh neural net and setup the optimizer with it
        #
        # Use a model with LSTM activations
        if lstm:
            self.model_to_use = self.getmodel_lstm(batchsize)
            self.optimizer.setup(self.model_to_use)
        else:
            self.model_to_use = self.getmodel_eeg()
            self.optimizer.setup(self.model_to_use)

        # Get EEG data to train with
        train_X_raw, train_Y = eeg_data.main.getdatasets_blink_ten()

        train_X = self.normalizevalues_eeg(train_X_raw)

        epochcount = 0
        datasize = train_X.shape[0]

        epoch_sums = []
        epoch_loss = []
        epochs = []

        volatile = False

        outputpath = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/" + self.outputfile + ".txt"
        sys.stdout = Logger(outputpath)
        #        for layer in self.model_to_use._children:
        #            print(layer[0], layer[1].W.shape)

        for epoch in range(200):
            t1 = datetime.now()
            epochcount += 1
            print('epoch %d' % (epoch + 1))
            sum_accuracy = 0.0
            sum_loss = 0.0

            y_batch_prev = Variable(np.zeros([batchsize, 14]), volatile=volatile)

            for i in range(0, datasize - batchsize):
                # Get current, previous and next sample from EEG recording to send through the net
                x_batch_curr = np.asarray(train_X[i:i + batchsize]).astype(np.float32).reshape(-1, batchsize)

                # Get current, previous and next sample from target output ready
                y_batch_curr = np.asarray(train_Y[i:i + batchsize]).astype(np.float32).reshape(-1, batchsize)

                self.optimizer.zero_grads()

                # Use the LSTM neural network
                if lstm:
                    x_batch_curr_np = np.mean(x_batch_curr, axis=0).reshape(-1, batchsize)
                    c = Variable(y_batch_curr, volatile=volatile)
                    accuracy, loss, c, y_batch_prev = self.forward_lstm(c, x_batch_curr_np, y_batch_prev)

                else:
                    x_batch_curr = x_batch_curr.reshape(1, -1)
                    y_batch_curr = y_batch_curr.astype(int32)
                    accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=False)

                sum_accuracy += accuracy.data
                sum_loss += abs(loss.data)

                output.data = output.data.reshape(1, -1)

                loss.backward()
                self.optimizer.update()

            t2 = datetime.now()
            print("\tAccuracy of training: ", sum_accuracy / datasize)
            print("\tLoss of training: ", sum_loss / datasize)
            time_diff = t2 - t1
            print("\tSeconds for epoch: ", time_diff.seconds)

            epoch_sums.append(sum_accuracy / datasize)
            epoch_loss.append(sum_loss / datasize)
            epochs.append(epoch)

            self.optimizer.lr += 0.00005

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
        self.model_to_use = self.getmodel_eeg()
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

    def gettargetdataset(self, target_state, size):
        return np.full((size, 1), target_state, dtype=int)

    def train_blink_ten(self, epochs_total, savemodel=True):
        train_X, train_Y = eeg_data.main.getdatasets_blink_ten()
        samples_lenth = train_X.__len__()
        for i in range(samples_lenth):
            print(train_X[i], " ", train_Y[i])

    def train_timefreq(self, savemodel=True):

        # Get raw data
        train_X_raw, y = eeg_data.main.getdatasets_eeg()

        # Get fft values
        # t.shape = 320
        # fft_y.shape = 640
        # spectro_data = [f, t, Sxx]
        #   f.shape = 321
        #   t.shape = 513
        #   sxx.shape = 321, 513
        # csd_data = [f, Pxy]
        #   f.shape = 321
        #   pxy.shape = 321

        t, fft_y, spectro_data, csd_data = eeg_data.main.getfft(train_X_raw)

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

    def test_eeg_sample(self):
        test_X, test_Y = eeg_data.main.getdatasets_blink_ten()

        datasize = test_X.shape[0]

        sum_accuracy = 0.0
        sum_loss = 0.0

        sample_preds = []
        batchsize = 32

        # Check if there is a saved model to use and load it if there is
        print(self.get_savedrnn())
        if self.get_savedrnn() != -1:
            self.model_to_use = self.get_savedrnn()
            self.optimizer.setup(self.model_to_use)
        else:
            self.model_to_use = pickle.load(open("C:/Users/SourabhKatti/Documents/engine/mozart/models/my.model", "rb"))
            self.optimizer.setup(self.model_to_use)

        for i in range(2, datasize - batchsize):
            x_batch_prev = np.asarray(test_X[i - 1:i - 1 + batchsize]).astype(np.float32).reshape(1, -1)
            x_batch_prev2 = np.asarray(test_X[i - 2:i - 2 + batchsize]).astype(np.float32).reshape(1, -1)
            x_batch_curr = np.asarray(test_X[i:i + batchsize]).astype(np.float32).reshape(1, -1)

            # Get current, previous and next sample from target output ready
            y_batch_prev2 = np.asarray(test_Y[i - 2:i - 2 + batchsize]).astype(np.int32)
            y_batch_prev = np.asarray(test_Y[i - 1:i - 1 + batchsize]).astype(np.int32)
            y_batch_curr = np.asarray(test_Y[i:i + batchsize]).astype(np.int32)

            self.optimizer.zero_grads()
            accuracy, loss, output = self.forward(x_batch_curr, y_batch_curr, volatile=True)

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

    def savernn(self):
        pickle.dump(self.model_to_use, open(self.model_path, 'wb'))

    def get_savedrnn(self):
        try:
            eeg_model = pickle.load(open(self.model_path, "rb"))
            return eeg_model
        except:
            return -1

    # Refine the predictions and plot the results along with the original EEG waveforms
    def plot_predictions(self, raw_data, predictions):
        indices = []
        preds = []
        raw_data_indices = []
        raw_data_values = []

        # Output preds have size [1, batchsize] so get the average of all the values
        for i, pred in enumerate(predictions):
            indices.append(i)
            preds.append(np.mean(pred[0]))
            print(i, np.mean(pred[0]))

        # Average the output prediction signal over a specified window
        pred_size = preds.__len__()

        avg_preds = []
        i = 0
        averaging_window = 1
        print("Averaging predictions with a %d sample window" % averaging_window)
        while i < pred_size - averaging_window:
            avg_pred = np.mean(preds[i:i + averaging_window])
            avg_preds.append(avg_pred)
            i += 1
        preds[0:pred_size - averaging_window] = avg_preds
        min_pred = np.min(avg_preds)
        max_pred = np.max(avg_preds)
        preds_norm = np.divide(preds, max_pred)
        preds_norm *= 2

        # Get raw EEG data to graph
        for e, value in enumerate(raw_data):
            raw_data_indices.append(e)
            raw_data_values.append(value)

        predictions = np.asarray(preds_norm).reshape(-1, 1)
        raw_data_values = np.asarray(raw_data_values).reshape(-1, 14)
        raw_data_indices = np.asarray(raw_data_indices).reshape(-1, 1)
        raw_data_shape = raw_data_values.shape

        # Plot the predictions and raw EEG signals
        plt.clf()
        predsfile = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/" + self.outputfile + "-preds.png"
        predsfig = plt.figure(2)
        predfigsb = predsfig.add_subplot(211)
        predfigsb.plot(indices, predictions)
        predfigsb.set_ylim([np.min(predictions), np.max(predictions)])

        actualfigdb = predsfig.add_subplot(212)
        data_index = 0
        while data_index < raw_data_shape[1]:
            actualfigdb.plot(raw_data_indices, raw_data_values[:, data_index])
            data_index += 1
        actualfigdb.set_ylim([np.min(raw_data_values), np.max(raw_data_values)])
        predsfig.show()
        predsfig.savefig(predsfile)


        # Setup an RNN using chainer library


eeg_learner = eeg_learner()

# training_data = eeg_learner.train_blink_ten(10)


raw_data = eeg_learner.train_timeonly()
test_X, predictions = eeg_learner.test_eeg_sample()
eeg_learner.plot_predictions(test_X, predictions)


# eeg_learner.train_eyes_open_close()
# test_input, predictions = eeg_learner.test_eye_states_model()
# eeg_learner.plot_predictions(test_input, predictions)


# eeg_learner.train_timefreq()
