import eeg_data.main
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
from scipy import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


def forward(x_batch_prev, x_batch_curr, x_batch_next, y_batch_prev, y_batch_curr, y_batch_next, volatile=False):
    accuracy = 0
    loss = 0

    current_sample = Variable(x_batch_curr, volatile=volatile)
    next_sample = Variable(x_batch_next, volatile=volatile)
    previous_sample = Variable(x_batch_prev, volatile=volatile)

    y_batch_curr = np.asarray(y_batch_curr).reshape(1, -1)
    current_output = Variable(y_batch_curr, volatile=volatile)
    next_output = Variable(y_batch_next, volatile=volatile)
    previous_output = Variable(y_batch_prev, volatile=volatile)

    h1_current = F.sigmoid(eeg_model.x_h1(current_sample))
    h1_previous = F.sigmoid(eeg_model.x_h1(previous_sample))
    h1_next = F.sigmoid(eeg_model.x_h1(next_sample))
    h1_diff_previous = h1_current - h1_previous
    h1_diff_next = h1_next - h1_current

    h2_current = F.sigmoid(eeg_model.h1_h2(h1_current))
    h2_diff_n = F.sigmoid(eeg_model.h1_h2(h1_diff_next))
    h2_diff_p = F.sigmoid(eeg_model.h1_h2(h1_diff_previous))
    h2_diff_next = h2_diff_n - h2_current
    h2_diff_previous = h2_current - h2_diff_p

    h3_current = F.sigmoid(eeg_model.h2_h3(h2_current))
    h3_diff_p = F.sigmoid(eeg_model.h2_h3(h2_diff_previous))
    h3_diff_n = F.sigmoid(eeg_model.h2_h3(h2_diff_next))
    h3_diff_next = h3_diff_n - h3_current
    h3_diff_previous = h3_current - h3_diff_p

    h4_current = F.sigmoid(eeg_model.h3_h4(h3_current))
    h4_diff_previous = F.sigmoid(eeg_model.h3_h4(h3_diff_previous))
    h4_diff_next = F.sigmoid(eeg_model.h3_h4(h3_diff_next))
    h4_diff = h4_diff_next * h4_diff_previous
    h4 = h4_current + h4_diff

    y = eeg_model.h4_y(h4)


    loss = F.sigmoid_cross_entropy(y, current_output)
    accuracy = F.accuracy(y, current_output)

    return accuracy, loss, y


def train_eeg_sample():
    train_plot = plt.figure()
    plt1 = train_plot.add_subplot(211)
    plt2 = train_plot.add_subplot(212)

    # Get EEG data to train with
    train_X_raw, train_Y = eeg_data.main.getdatasets_eeg()

    train_X = normalizevalues_eeg(train_X_raw)

    epochcount = 0
    datasize = train_X.shape[0]

    epoch_sums = []
    epoch_loss = []
    epochs = []

    outputpath = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/" + outputfile + ".txt"
    sys.stdout = Logger(outputpath)
    for layer in eeg_model._get_sorted_funcs():
        print(layer[0], layer[1].W.shape)

    for epoch in range(2):
        epochcount += 1
        print('epoch %d' % epoch)
        sum_accuracy = 0.0
        sum_loss = 0.0
        for i in range(1, datasize - 1):
            # Get current, previous and next sample from EEG recording to send through the net
            x_batch_prev = np.asarray(train_X[i - 1]).astype(np.float32).reshape(1, -1)
            x_batch_next = np.asarray(train_X[i + 1]).astype(np.float32).reshape(1, -1)
            x_batch_curr = np.asarray(train_X[i]).astype(np.float32).reshape(1, -1)

            # Get current, previous and next sample from target output ready
            y_batch_prev = np.asarray(train_Y[i - 1]).astype(np.int32)
            y_batch_next = np.asarray(train_Y[i + 1]).astype(np.int32)
            y_batch_curr = np.asarray(train_Y[i]).astype(np.int32)

            optimizer.zero_grads()
            accuracy, loss, output = forward(x_batch_prev, x_batch_curr, x_batch_next, y_batch_prev, y_batch_curr,
                                             y_batch_next, volatile=False)

            sum_accuracy += accuracy.data
            sum_loss += loss.data

            loss.backward()
            optimizer.update()

        print("\tAccuracy of training: ", sum_accuracy / datasize)
        print("\tLoss of training: ", sum_loss / datasize)

        epoch_sums.append(sum_accuracy / datasize)
        epoch_loss.append(sum_loss / datasize)
        epochs.append(epoch)

    plt1.plot(epochs, epoch_sums)
    plt2.plot(epochs, epoch_loss)
    pltimage = outputpath + '.png'
    train_plot.show()
    train_plot.savefig(pltimage)
    return train_X


def normalizevalues_eeg(raw_eeg):
    eeg_norm = []
    x, y = raw_eeg.shape

    sampleindex = 0

    while sampleindex < x:
        raw_sample = raw_eeg[sampleindex]
        max_value = np.max(raw_sample)
        norm_sample = []
        for value in raw_sample:
            norm_sample.append(float(value) / float(max_value))
        eeg_norm.append(norm_sample)
        sampleindex += 1
    eeg_norm = np.asarray(eeg_norm)

    return eeg_norm


def test_eeg_sample():
    test_X, test_Y = eeg_data.main.getdatasets_eeg()

    datasize = test_X.shape[0]

    sum_accuracy = 0.0
    sum_loss = 0.0

    sample_preds = []
    for i in range(1, datasize - 1):
        x_batch_prev = np.asarray(test_X[i - 1]).astype(np.float32).reshape(1, -1)
        x_batch_next = np.asarray(test_X[i + 1]).astype(np.float32).reshape(1, -1)
        x_batch_curr = np.asarray(test_X[i]).astype(np.float32).reshape(1, -1)

        # Get current, previous and next sample from target output ready
        y_batch_prev = np.asarray(test_Y[i - 1]).astype(np.int32)
        y_batch_next = np.asarray(test_Y[i + 1]).astype(np.int32)
        y_batch_curr = np.asarray(test_Y[i]).astype(np.int32)

        optimizer.zero_grads()
        accuracy, loss, output = forward(x_batch_prev, x_batch_curr, x_batch_next, y_batch_prev, y_batch_curr,
                                         y_batch_next, volatile=True)

        sum_accuracy += accuracy.data
        sum_loss += loss.data

        loss.backward()
        optimizer.update()

        sample_preds.append(output.data)

    min_preds = np.min(sample_preds)
    normalized_preds = []
    for value in sample_preds:
        normalized_preds.append(np.divide(value, min_preds).astype(float))
    # max_preds = np.max(normalized_preds)
    # normalized_max_preds = np.divide(normalized_preds, max_preds)

    return normalized_preds


def plot_predictions(raw_data, predictions):
    indices = []
    preds = []
    raw_data_indices = []
    raw_data_values = []

    for i, pred in enumerate(predictions):
        indices.append(i)
        preds.append(pred[0][0])
        print(i, pred[0][0])

    for e, value in enumerate(raw_data):
        raw_data_indices.append(e)
        raw_data_values.append(value)


    predictions = np.asarray(preds).reshape(-1, 1)
    raw_data_values = np.asarray(raw_data_values).reshape(-1, 14)
    raw_data_indices = np.asarray(raw_data_indices).reshape(-1, 1)
    raw_data_shape = raw_data_values.shape

    plt.clf()
    predsfile = "C:/Users/SourabhKatti/Documents/engine/mozart/logs/" + outputfile + "-preds.png"
    predsfig = plt.figure(2)
    predfigsb = predsfig.add_subplot(211)
    predfigsb.plot(indices, predictions)
    predfigsb.set_ylim([np.min(predictions), np.max(predictions)])

    actualfigdb = predsfig.add_subplot(212)
    data_index = 0
    while data_index < raw_data_shape[1]:
        actualfigdb.plot(raw_data_indices, raw_data_values[:,data_index])
        data_index += 1
    actualfigdb.set_ylim([np.min(raw_data_values), np.max(raw_data_values)])
    predsfig.show()
    predsfig.savefig(predsfile)


# Setup an RNN using chainer library

eeg_model = FunctionSet(
        x_h1=F.Linear(14, 64),
        h1_h2=F.Linear(64, 256),
        h2_h3=F.Linear(256, 256),
        h3_h4=F.Linear(256, 4),
        h4_y=F.Linear(4, 1),
)
optimizer = optimizers.SGD()
optimizer.setup(eeg_model)

# Setup filename to log output to
outputfile = str(time.time())

raw_data = train_eeg_sample()
predictions = test_eeg_sample()
plot_predictions(raw_data, predictions)
