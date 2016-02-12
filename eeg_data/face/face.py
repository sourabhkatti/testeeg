import eeg_data.face.edf as edf


# noinspection PyMethodMayBeStatic



def get_blink_twice_ds():
    path_blink_twice_csv = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/blink-twice/blink-twice-1-03.02.16.19.40.57.csv"
    path_blink_twice_target = "C:/testeeg/testeeg/eeg_data/face/edf/converted_csv/blink-twice/blink-twice-1-03.02.16.19.40.57.txt"

    gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s = edf.importfile(path_blink_twice_csv)
    train_y = edf.get_targets_by_ms(path_blink_twice_target, time_ms)

    return [gyro_x, gyro_y, electrodes, sample_numbers, time_ms, time_s], train_y
