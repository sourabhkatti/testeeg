import numpy as np
import os
import re


# sample number = 1

# AF3 = 3
# F7 = 4
# F3 = 5
# FC5 = 6
# T7 = 7
# P7 = 8
# O1 = 9
# O2 = 10
# P8 = 11
# T8 = 12
# FC6 = 13
# F4 = 14
# F8 = 15
# AF4 = 16

# gyro_x = 17
# gyro_y = 18

# time_s = 21
# time_ms = 22


## import data from an csv file.
# Input takes in a path to a directory of csv files
def importfolder(path, count):
    raw_edfs = []
    gyro_x = []
    gyro_y = []
    electrodes = []
    sample_numbers = []
    time_ms = []
    time_s = []

    print(path)

    # Return if there are less files in the folder than the requested count
    files = os.listdir(path)
    if files.__len__() < count:
        print("No files found")
        return

    for file in files:
        path_to_file = path + '/' + file
        csvmatch = re.search("csv", path_to_file)
        if csvmatch:
            print(path_to_file)
            raw_edf = np.loadtxt(path_to_file, dtype=str, delimiter=',', skiprows=1)
            raw_edfs.append(raw_edf)


def importfile(path):
    raw_edfs = []
    gyro_x = []
    gyro_y = []
    electrodes = []
    sample_numbers = []
    time_ms = []
    time_s = []

    raw_edf = np.loadtxt(path, dtype=str, delimiter=', ', skiprows=1)
    sanitized_edf = sanitizeedf(raw_edf)

    electrodes = sanitized_edf[:, 2:16]

    sample_numbers = sanitized_edf[:, 0]

    time_ms = sanitized_edf[:, 22]
    time_s = sanitized_edf[:, 21]

    gyro_x = sanitized_edf[:, 17]
    gyro_y = sanitized_edf[:, 18]

    return gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s

# Clean out the special characters in an edf and convert to float
def sanitizeedf(raw_edf):
    raw_edf = np.asarray(raw_edf)
    sanitized_edf = np.zeros(raw_edf.shape, dtype=float)
    sample_index = 0
    for sample in raw_edf:
        sanitized_sample = np.zeros(sample.shape, dtype=float)
        channel_index = 0
        for channel in sample:
            sanitized_sample[channel_index] = np.asarray(re.sub("[a-z\']", string=channel, repl="")).astype(float)
            channel_index += 1
        sanitized_edf[sample_index] = sanitized_sample
        sample_index += 1
    return sanitized_edf




# Return a list of targets based on a provided text file
def get_targets_by_ms(path_blink_twice_target, time_ms):
    # Load up values of ms to compare to
    target_counts = np.loadtxt(path_blink_twice_target, dtype=float, delimiter=' ')
    target_index = 0

    samples = time_ms.__len__()
    sample_index = 0

    ms_sum = 0

    target_output = []

    # For each sample in time_ms array, write a 0 as long as the target_counts have not been reached.
    # If a target count is reached, write a 1 for the next sample

    # Add a 0 to help the array sizes match up later
    target_output.append(0)

    while sample_index < samples - 1:
        time_diff = time_ms[sample_index + 1] - time_ms[sample_index]

        if time_diff < 0:
            time_diff += 1000.0

        ms_sum += time_diff

        try:
            if ms_sum > target_counts[target_index]:
                target_output.append(1)
                target_index += 1
            else:
                target_output.append(0)
        except:
            target_output.append(0)

        sample_index += 1

    return np.asarray(target_output).astype(float)
