import numpy as np
import pandas as pd

def sliding_win(series, window_size):
    subsequences = []
    for i in range(len(series) - window_size + 1):
        subsequences.append(series[i:i + window_size])
    return np.array(subsequences)

def sliding_win_target(series, window_size, win_out):
    X, y = [], []
    for i in range(len(series) - window_size - win_out + 1):
        win = series[i:i + window_size]
        target = series[i + window_size : i + window_size + win_out]
        X.append(win)
        y.append(target)
    return np.array(X), np.array(y)

def sliding_win_target_offst(series, window_size, win_out, offset=0):
    """
    Partitions a time series into sliding windows with support for a target window and an offset.
    
    Parameters:
        series (array-like): The time series to be partitioned.
        window_size (int): The size of each input window.
        win_out (int): The size of the target window.
        offset (int): The starting index offset (default: 0).
    
    Returns:
        X (np.ndarray): Input windows.
        y (np.ndarray): Target windows.
    """
    X, y = [], []
    # Ensure offset does not exceed series length
    offset = max(0, offset)
    
    for i in range(offset, len(series) - window_size - win_out + 1):
        win = series[i:i + window_size]
        target = series[i + window_size:i + window_size + win_out]
        X.append(win)
        y.append(target)
    
    return X,y

# def sliding_win_multiple_offsets_list(series, window_size, win_out, num_offsets):
#     """
#     Iterates the sliding window procedure over a specific number of offsets and returns results as a list.
    
#     Parameters:
#         series (array-like): The time series to be partitioned.
#         window_size (int): The size of each input window.
#         win_out (int): The size of the target window.
#         num_offsets (int): The number of offsets to iterate over.
    
#     Returns:
#         results (list): A list where each entry is a tuple (X, y) for a specific offset.
#     """
#     results = []
    
#     for offset in range(num_offsets):
#         offset_X, offset_y = sliding_win_target(series, window_size, win_out, offset=offset)
#         results.append((offset_X, offset_y))
    
#     return results


def sliding_win_target_multiple_offsets_list(series, window_size, win_out, num_offsets):
    """
    Iterates the sliding window procedure over a specific number of offsets and returns results as a list.
    
    Parameters:
        series (array-like): The time series to be partitioned.
        window_size (int): The size of each input window.
        win_out (int): The size of the target window.
        num_offsets (int): The number of offsets to iterate over.
    
    Returns:
        results X(list): A list where each entry contains the training window for incremental offsets.
        results y(list): A list where each entry contains the training target(win_out) for incremental offsets.

    """
    results_X = []
    results_y = []
    for offset in range(num_offsets):
        offset_X, offset_y = sliding_win_target_offst(series, window_size, win_out, offset=offset)
        results_X.append(offset_X)
        results_y.append(offset_y)
    
    return results_X, results_y

def sliding_win_cluster_aware(series, window_size_cluster, window_size_pred, win_out_pred):
    assert window_size_cluster >= window_size_pred
    X_cluster, X_pred, y = [], [], []
    for i in range(len(series) - window_size_cluster - win_out_pred + 1):
        cluster = series[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = series[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster)
        X_pred.append(pred_window)
        y.append(target)
    return np.array(X_cluster), np.array(X_pred), np.array(y)

""" Same of the other, just partition also the timestamps """
def sliding_win_timestamps(series, timestamps, window_size):
    assert len(series) == len(timestamps)
    subsequences = []
    subs_ts = []
    for i in range(len(series) - window_size + 1):
        subsequences.append(series[i:i + window_size])
        subs_ts.append(timestamps[i : i + window_size])
    return np.array(subsequences), np.array(subs_ts)

def sliding_win_target_timestamps(series, time_series_timestamps, window_size, win_out):
    assert len(series) == len(time_series_timestamps)
    X, y = [], []
    X_tim, y_tim = [], []
    for i in range(len(series) - window_size - win_out + 1):
        win = series[i:i + window_size]
        win_tim = time_series_timestamps[i:i + window_size]
        target = series[i + window_size : i + window_size + win_out]
        target_timestamps = time_series_timestamps[i + window_size : i + window_size + win_out]
        X.append(win)
        y.append(target)
        X_tim.append(win_tim)
        y_tim.append(target_timestamps)
    return np.array(X),np.array(X_tim), np.array(y), np.array(y_tim)

def sliding_win_clust_aw_target_timestamps(series, time_series_timestamps, window_size_cluster, window_size_pred, win_out_pred):
    assert window_size_cluster > window_size_pred
    X_cluster, X_pred, y = [], [], []
    X_cluster_ts, X_pred_ts, y_ts = [], [], []
    for i in range(len(series) - window_size_cluster - win_out_pred + 1):
        # Generate cluster data
        cluster = series[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = series[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster)
        X_pred.append(pred_window)
        y.append(target)
        # Generate time series data
        cluster_ts = time_series_timestamps[i:i + window_size_cluster]      
        pred_window_ts = cluster_ts[-window_size_pred:]
        target_ts = time_series_timestamps[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster_ts.append(cluster_ts)
        X_pred_ts.append(pred_window_ts)
        y_ts.append(target_ts)
    return np.array(X_cluster), np.array(X_cluster_ts), np.array(X_pred), np.array(X_pred_ts), np.array(y), np.array(y_ts)

def partition_series(series, elements_per_partition):
    """
    Partitions a series into equal-length parts based on the number of elements per partition.
    
    Parameters:
        series (list or numpy.ndarray): The series to be partitioned.
        elements_per_partition (int): The number of elements in each partition.
        
    Returns:
        list of numpy.ndarray: A list containing the partitioned parts as numpy arrays.
    """
    series = np.array(series)  # Ensure the input is a numpy array
    series_length = len(series)
    
    # Calculate the number of partitions and trim the excess elements
    num_partitions = series_length // elements_per_partition
    trimmed_length = num_partitions * elements_per_partition
    series = series[:trimmed_length]
    
    # Split the series into partitions
    return [series[i * elements_per_partition:(i + 1) * elements_per_partition] 
            for i in range(num_partitions)]


def partition_series_multivariate(series, elements_per_partition):

    #series = np.array(series)  # Ensure the input is a numpy array
    series_length = len(series)
    
    # Calculate the number of partitions and trim the excess elements
    num_partitions = series_length // elements_per_partition
    trimmed_length = num_partitions * elements_per_partition
    series = series[:trimmed_length]
    
    # Split the series into partitions
    return [series.iloc[i * elements_per_partition:(i + 1) * elements_per_partition].values.reshape(-1) 
            for i in range(num_partitions)]

def sliding_windows_multivariate(df_series, window_size):
    rows = df_series.to_numpy()
    subsequences = []
    for i in range(len(rows) - window_size + 1):
        sub = rows[i:i + window_size].reshape(-1)
        subsequences.append(sub)
    return np.array(subsequences)

def sliding_windows_multivariate_target(df_series, window_size, target_column, win_out):
    rows = df_series.to_numpy()
    tc = df_series[target_column].to_numpy().reshape(-1)
    subsequences = []
    target_sequences = []
    for i in range(len(rows) - window_size + 1):
        sub = rows[i:i + window_size].reshape(-1)
        target = tc[i + window_size:i + window_size + win_out]
        subsequences.append(sub)
        target_sequences.append(target)
    return np.array(subsequences), np.array(target_sequences)

def sliding_win_cluster_aware_multivariate(df_series, target_column, window_size_cluster, window_size_pred, win_out_pred):
    assert window_size_cluster >= window_size_pred
    X_cluster, X_pred, y = [], [], []
    rows = df_series.to_numpy()
    target_values = df_series[target_column].to_numpy().reshape(-1)
    for i in range(len(rows) - window_size_cluster - win_out_pred + 1):
        cluster = rows[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = target_values[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster.reshape(-1))
        X_pred.append(pred_window.reshape(-1))
        y.append(target)
    return np.array(X_cluster), np.array(X_pred), np.array(y)


def custom_mape(y_true, y_pred):
    act_resh = np.array(y_true).reshape(-1)
    for_resh = np.array(y_pred).reshape(-1)
    sum = 0
    for act, forec in zip(act_resh, for_resh):
        if act > 0.0:
            sum += abs((act - forec) / act)
        else:
           sum += abs((act - forec))
    return sum / len(act_resh) * 100

if __name__ == "__main__":
    # Sample data
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [2, 3, 4, 5, 6, 7, 8, 9],
        'target': [10, 20, 30, 40, 50, 60, 70, 80]
    }
    df_series = pd.DataFrame(data)
    print(df_series)
    print(sliding_windows_multivariate(df_series,3))
    print("Cluster aware partitioning")
    win_clust, win_tree, pred = sliding_win_cluster_aware_multivariate(df_series,"target", 3, 3, 1)
    print("Clusters")
    print(win_clust)
    print("Targets")
    print(pred)
    # exit(1)
    seq = np.array([i for i in range(1,20)])
    offset = 1
    series_offst = sliding_win_target_offset(seq, 4, 1, offset)
    print(series_offst)
    exit(1)
    seq_ts = np.array([i for i in range(0,15)])
    print("Test clustering")
    win_cluster, win_pred, target = sliding_win_cluster_aware(seq, 5, 3, 1)
    print(win_cluster)
    print(win_pred)
    print(target)
    print("Test clustering with timestamps")
    win_cluster, win_cluster_ts, win_pred, win_pred_ts, target, target_ts = sliding_win_clust_aw_target_timestamps(seq, seq_ts, 5, 3, 1)
    print("Showing win cluster ts")
    for x,y in zip(win_cluster, win_cluster_ts):
        print(x)
        print(y)
    print("Showing win pred ts")
    for x,y in zip(win_pred, win_pred_ts):
        print(x)
        print(y)
    print("Showing targets")
    for x,y in zip(target, target_ts):
        print(x)
        print(y)
    