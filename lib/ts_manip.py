import numpy as np

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


if __name__ == "__main__":
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
    