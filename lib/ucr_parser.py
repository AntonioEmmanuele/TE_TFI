import numpy as np
import os
from tqdm import tqdm
import math
from .ts_manip import aggregate_with_mean_time_window
def ucr_clean_from_anomalies(time_series, training_start):
    return time_series[0 : training_start]

def ucr_get_training_starting_point(ts_file_name):
    # Remove .txt and split around the _, the timeseries start train is the last number.
    splitted_list = ts_file_name[:-4].split("_")
    return int(splitted_list[-3])

# def ucr_decicide_window_size(ts_len, thd = 10000):
#     return 0

def ucr_process_all_ts(in_path, out_path, size_thd = 20000, aggr_win_size = 5, disable_tqdm = False):
    file_list = os.listdir(in_path)
    for ts in tqdm(file_list, desc="Processing UCR TS", disable = disable_tqdm):
        # Compose the path
        ts_path = os.path.join(in_path, ts)
        ts_out_path = os.path.join(out_path, ts)
        # Get the end of the training
        end_train = ucr_get_training_starting_point(ts)
        # Load the txt
        series = np.loadtxt(ts_path)
        # Compose the function
        series = series[0:end_train]
        # If greater than the max number, aggregate
        if len(series) > size_thd:
            series = aggregate_with_mean_time_window(series = series, window_size = aggr_win_size)
        np.savetxt(ts_out_path, series)
