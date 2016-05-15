import eeg_data.face.edf as edf
import os
import numpy as np
import random


# noinspection PyMethodMayBeStatic



def get_blink_twice_ds():
    path_blink_twice_csv = "C:/users/sourabhkatti/documents/engine/eeg_data/face/edf/converted_csv/blink-twice/blink-twice-1-03.02.16.19.40.57.csv"
    path_blink_twice_target = "C:/users/sourabhkatti/documents/engine/eeg_data/face/edf/converted_csv/blink-twice/blink-twice-1-03.02.16.19.40.57.txt"

    gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_blink_twice_csv)
    train_y = edf.get_targets_by_ms(path_blink_twice_target, time_ms)

    return [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s], train_y


def get_blink_ten_ds():
    path_blink_ten_csv = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/blink-ten/blink_10.csv"
    path_blink_ten_target = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/blink-ten/blink_10_targets.txt"
    gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_blink_ten_csv)
    train_y = edf.get_targets_by_ms(path_blink_ten_target, time_ms)

    return [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s], train_y


def get_eyes_open_ds():
    path_eyes_open_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-open"
    files = os.listdir(path_eyes_open_csv)
    electrodes_all = []
    gyro_x_all = []
    gyro_y_all = []
    sample_numbers_all = []
    time_ms_all = []
    time_s_all = []

    print("Getting eye-open datasets...")
    for file in files:
        print('\tLoading ' + file)
        path_file = path_eyes_open_csv + '/' + file
        gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_file)
        electrodes_all = addfiles(electrodes_all, electrodes)
        gyro_x_all = addfiles(gyro_x_all, gyro_x)
        gyro_y_all = addfiles(gyro_y_all, gyro_y)
        sample_numbers_all = addfiles(sample_numbers_all, sample_numbers)
        time_ms_all = addfiles(time_ms_all, time_ms)
        time_s_all = addfiles(time_s_all, time_s)

    return [gyro_x_all, gyro_y_all, electrodes_all, sample_numbers_all, time_ms_all, time_s_all]


def get_eyes_closed_ds():
    path_eyes_closed_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-closed"
    files = os.listdir(path_eyes_closed_csv)
    electrodes_all = []
    gyro_x_all = []
    gyro_y_all = []
    sample_numbers_all = []
    time_ms_all = []
    time_s_all = []

    print("\nGetting eye-closed datasets...")
    for file in files:
        print("\tLoading " + file)
        path_file = path_eyes_closed_csv + '/' + file
        gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_file)
        electrodes_all = addfiles(electrodes_all, electrodes)
        gyro_x_all = addfiles(gyro_x_all, gyro_x)
        gyro_y_all = addfiles(gyro_y_all, gyro_y)
        sample_numbers_all = addfiles(sample_numbers_all, sample_numbers)
        time_ms_all = addfiles(time_ms_all, time_ms)
        time_s_all = addfiles(time_s_all, time_s)

    return [gyro_x_all, gyro_y_all, electrodes_all, sample_numbers_all, time_ms_all, time_s_all]


# Merge multiple CSV files together through indices
def addfiles(master, newfile):
    newfilenp = np.asarray(newfile)
    newfileshape = newfilenp.shape
    for index in range(newfileshape[0]):
        master.append(newfile[index])
    return master


def get_blink_ten_ds():
    path_blink_ten_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/blink-ten/blink_10.csv"
    path_blink_ten_targets = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/blink-ten/blink_10_targets.txt"

    gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_blink_ten_csv)

    train_y = edf.get_targets_by_ms(path_blink_ten_targets, time_ms)

    return [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s], train_y


def get_eye_states_test_ds():
    path_eyes_closed_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-closed"
    path_eyes_open_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-open"
    electrodes = []

    eyes_open_ds = os.listdir(path_eyes_open_csv)
    random_eyes_open_ds = random.choice(eyes_open_ds)
    path_random_eyes_open_ds = path_eyes_open_csv + '/' + random_eyes_open_ds

    eyes_closed_ds = os.listdir(path_eyes_closed_csv)
    random_eyes_closed_ds = random.choice(eyes_closed_ds)
    path_random_eyes_closed_ds = path_eyes_closed_csv + '/' + random_eyes_closed_ds

    gyro_x_open, gyro_y_open, electrodes_open, sample_numbers_open, time_ms_open, time_s_open = edf.importfile(
        path_random_eyes_open_ds)
    gyro_x_closed, gyro_y_closed, electrodes_closed, sample_numbers_closed, time_ms_closed, time_s_closed = edf.importfile(
        path_random_eyes_closed_ds)

    dataset_open_size = electrodes_open.__len__()
    for i in range(dataset_open_size):
        electrodes.append(electrodes_open[i])

    dataset_closed_size = electrodes_closed.__len__()
    for i in range(dataset_closed_size):
        electrodes.append(electrodes_closed[i])

    return electrodes


def get_eyes_open_ds():
    path_eyes_open_csv = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/eyes-open"
    files = os.listdir(path_eyes_open_csv)
    electrodes_all = []
    gyro_x_all = []
    gyro_y_all = []
    sample_numbers_all = []
    time_ms_all = []
    time_s_all = []

    print("Getting eye-open datasets...")
    for file in files:
        print('\tLoading ' + file)
        path_file = path_eyes_open_csv + '/' + file
        gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_file)
        electrodes_all = addfiles(electrodes_all, electrodes)
        gyro_x_all = addfiles(gyro_x_all, gyro_x)
        gyro_y_all = addfiles(gyro_y_all, gyro_y)
        sample_numbers_all = addfiles(sample_numbers_all, sample_numbers)
        time_ms_all = addfiles(time_ms_all, time_ms)
        time_s_all = addfiles(time_s_all, time_s)

    return [gyro_x_all, gyro_y_all, electrodes_all, sample_numbers_all, time_ms_all, time_s_all]


def get_eyes_closed_ds():
    path_eyes_closed_csv = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/eyes-closed"
    files = os.listdir(path_eyes_closed_csv)
    electrodes_all = []
    gyro_x_all = []
    gyro_y_all = []
    sample_numbers_all = []
    time_ms_all = []
    time_s_all = []

    print("\nGetting eye-closed datasets...")
    for file in files:
        print("\tLoading " + file)
        path_file = path_eyes_closed_csv + '/' + file
        gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_file)
        electrodes_all = addfiles(electrodes_all, electrodes)
        gyro_x_all = addfiles(gyro_x_all, gyro_x)
        gyro_y_all = addfiles(gyro_y_all, gyro_y)
        sample_numbers_all = addfiles(sample_numbers_all, sample_numbers)
        time_ms_all = addfiles(time_ms_all, time_ms)
        time_s_all = addfiles(time_s_all, time_s)

    return [gyro_x_all, gyro_y_all, electrodes_all, sample_numbers_all, time_ms_all, time_s_all]


def addfiles(master, newfile):
    newfilenp = np.asarray(newfile)
    newfileshape = newfilenp.shape
    for index in range(newfileshape[0]):
        master.append(newfile[index])
    return master


def get_eye_states_test_ds():
    path_eyes_closed_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-closed"
    path_eyes_open_csv = "C:/Users/SourabhKatti/Documents/engine/eeg_data/face/edf/converted_csv/eyes-open"
    electrodes = []

    eyes_open_ds = os.listdir(path_eyes_open_csv)
    random_eyes_open_ds = random.choice(eyes_open_ds)
    path_random_eyes_open_ds = path_eyes_open_csv + '/' + random_eyes_open_ds

    eyes_closed_ds = os.listdir(path_eyes_closed_csv)
    random_eyes_closed_ds = random.choice(eyes_closed_ds)
    path_random_eyes_closed_ds = path_eyes_closed_csv + '/' + random_eyes_closed_ds

    gyro_x_open, gyro_y_open, electrodes_open, sample_numbers_open, time_ms_open, time_s_open = edf.importfile(
            path_random_eyes_open_ds)
    gyro_x_closed, gyro_y_closed, electrodes_closed, sample_numbers_closed, time_ms_closed, time_s_closed = edf.importfile(
            path_random_eyes_closed_ds)

    dataset_open_size = electrodes_open.__len__()
    for i in range(dataset_open_size):
        electrodes.append(electrodes_open[i])

    dataset_closed_size = electrodes_closed.__len__()
    for i in range(dataset_closed_size):
        electrodes.append(electrodes_closed[i])

    return electrodes
