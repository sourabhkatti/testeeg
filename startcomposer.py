import os
import mozart.main


def getworkingdirectory():
    workingdir = os.getcwd()
    workingdirectory = workingdir.replace('\\', "/")
    return workingdirectory


class Composer:
    def __init__(self):
        self.workingdirectory = getworkingdirectory()
        pass

    def run(self):
        themozart = mozart.main.eeg_learner(self.workingdirectory)

        # training_data = eeg_learner.train_timeonly()
        # raw_data, predictions = eeg_learner.test_eeg_sample()
        # eeg_learner.plot_predictions(raw_data, predictions)

        # eeg_learner.train_eyes_open_close()
        # test_input, predictions = eeg_learner.test_eye_states_model()
        # eeg_learner.plot_predictions(test_input, predictions)

        batchsize = 256
        themozart.train_blink_ten(batchsize, print_frequency_graph=True, savemodel=True)
        raw_data, predictions, targetoutputs = themozart.test_models(batchsize)
        themozart.plot_predictions(raw_data, predictions, targetoutputs, batchsize)
        pass


composer = Composer()
composer.run()
