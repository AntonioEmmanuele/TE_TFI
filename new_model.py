import numpy as np
import pandas as pd
from lib.ts_manip import sliding_win, sliding_win_target, sliding_win_target_multiple_offsets_list
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

def compute_error_metric(true_values, predicted_values, metric="absolute"):
    """
    Compute an error metric for each sample.
    
    Parameters:
        true_values (array-like): The ground truth values.
        predicted_values (array-like): The predicted values.
        metric (str): The error metric to compute ("absolute", "squared", "percentage", "log_loss", "zero_one").
        
    Returns:
        np.ndarray: The error for each sample.
    """

    true_values = np.array(true_values).reshape(-1)
    predicted_values = np.array(predicted_values).reshape(-1)
    if metric == "absolute":
        return np.abs(true_values - predicted_values)
    elif metric == "squared":
        return (true_values - predicted_values)**2
    elif metric == "percentage":
        return 100 * np.abs((true_values - predicted_values) / true_values)
    elif metric == "zero_one":
        return (true_values != predicted_values).astype(int)
    elif metric == "log_loss":
        return - (true_values * np.log(predicted_values) + (1 - true_values) * np.log(1 - predicted_values))
    else:
        raise ValueError("Unsupported metric. Choose from 'absolute', 'squared', 'percentage', 'log_loss', 'zero_one'.")
    

def generate_increasing_percentages(total=100, max_elements=10):
    """
    Generates a sequence of increasing percentages summing to a given total, 
    with a variable number of percentages decreasing in each iteration.
    
    Parameters:
        total (float): The total percentage to reach (default: 100).
        max_elements (int): The initial maximum number of percentages.
    
    Returns:
        list of lists: A list containing sequences of increasing percentages.
    """
    results = []
    for n in range(max_elements, 0, -1):  # Decreasing number of elements
        # Generate `n` random values
        random_values = np.random.rand(n)
        
        # Sort values in increasing order
        sorted_values = np.sort(random_values)
        
        # Normalize values to sum to `total`
        normalized_percentages = (sorted_values / np.sum(sorted_values)) * total
        
        # Append to results
        results.append(normalized_percentages.tolist())
    
    return results

def partition_list(lst, n):
    """
    Partitions a list into `n` equal parts while maintaining sequentiality.
    Discards any remaining elements if the list cannot be evenly divided.
    
    Parameters:
        lst (list): The list to be partitioned.
        n (int): The number of partitions.
    
    Returns:
        list of lists: A list containing `n` partitions.
    """
    # Calculate the size of each partition (truncate any remainder)
    partition_size = len(lst) // n
    
    # Slice the list into `n` partitions
    partitions = [lst[i * partition_size:(i + 1) * partition_size] for i in range(n)]
    
    return partitions

# Load the series from the provided file

file_path = './datasets/processed/209_UCR_Anomaly_Fantasia_19000_26970_27270.txt'
series = np.loadtxt(file_path)
train_perc = 0.7
train_size = int(train_perc * len(series))
train_series = series[0 : train_size]
test_series  = series[train_size : ]
num_trees = 50
w_s_new = 10
train_X, train_y = sliding_win_target_multiple_offsets_list(series = train_series, window_size = w_s_new, win_out = 1, num_offsets = num_trees)
min_size = len(train_X[num_trees - 1])
sliding_percentage = 0.2 
for i in range(num_trees):
    adj = len(train_X[i]) - (len(train_X[i])-min_size)
    train_X[i] = train_X[i][:adj]
    train_X[i] = [np.append(train_X[i][j], 0.0) for j in range(len(train_X[i]))]

    train_y[i] = train_y[i][:adj]
    #train_y[i] = partition_list(train_y[i], num_trees)[i]

    
# percentages = generate_increasing_percentages(1.0, max_elements = num_trees)
# print(percentages)
# exit(1)
perc = 0.6
trees = [DecisionTreeRegressor(max_depth = 20, random_state = 42) for i in range(num_trees)]
# t_x = train_X[0]
# t_y = train_y[0]
for i in range(num_trees):
    print(f"Fitting {i}")
    trees[i].fit(train_X[i], train_y[i])
    if i + 1 < num_trees:
        preds = trees[i].predict(train_X[i + 1])
        errors = compute_error_metric(true_values = train_y[i + 1], predicted_values = preds, metric =  "squared")
        for j in range(len(train_X[i + 1])):
            train_X[i + 1][j][len((train_X[i+1][j])) - 1] = errors[j]

    
# exit(1)
# Now predict the new time series.
test_X, test_y = sliding_win_target_multiple_offsets_list(series = test_series, window_size = w_s_new, win_out = 1, num_offsets = num_trees)
min_size = len(test_X[num_trees - 1])
for i in range(num_trees):
    adj = len(test_X[i]) - (len(test_X[i]) - min_size)
    test_X[i] = test_X[i][:adj]
    test_X[i] = [np.append(test_X[i][j], 0.0) for j in range(len(test_X[i]))]
    test_y[i] = test_y[i][:adj]


 
final_preds = []
for i in range(num_trees):
    preds = trees[i].predict(test_X[i])
    if i + 1 <  num_trees:
        errors = compute_error_metric(true_values = test_y[i], predicted_values = preds, metric =  "squared")
        for j in range(len(test_X[i + 1])):
            test_X[i + 1][j][len((test_X[i+1][j])) - 1] = errors[j]
    else:
        final_preds = preds

mse = mean_squared_error(y_true = test_y[num_trees - 1], y_pred = final_preds)

print(f"The MSE is {mse}")
class_train_X, class_train_Y = sliding_win_target(train_series, window_size=num_trees * w_s_new, win_out=1)
class_test_X, class_test_Y = sliding_win_target(test_series, window_size=num_trees * w_s_new, win_out=1)
dt = DecisionTreeRegressor(max_depth=20, random_state = 42)
dt.fit(class_train_X, class_train_Y)
preds_dt = dt.predict(class_test_X)
mse_dt = mean_squared_error(y_true = class_test_Y, y_pred = preds_dt)
print(f"DT MSE {mse_dt}")
rf = RandomForestRegressor(n_estimators = 50, max_depth = 20, random_state=42)
rf.fit(class_train_X, class_train_Y)
preds_rf =  rf.predict(class_test_X)
mse_rf = mean_squared_error(y_true = class_test_Y, y_pred = preds_rf)
print(f"RF MSE {mse_rf}")