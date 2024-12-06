from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lib.ts_manip import partition_series, sliding_win, compute_l2_norm, sliding_win_target
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def compute_av_l2(set_of_samples, single_sample):
    values = []
    for s in set_of_samples:
        values.append(compute_l2_norm(s,single_sample))
    return np.mean(values)

def find_partition(set_of_sets, sample):
    means = []
    for s in set_of_sets:
        means.append(compute_av_l2(s,sample))
    return np.argmin(means)

path = "./datasets/processed/059_UCR_Anomaly_DISTORTEDgait1_20000_38500_38800.txt"
series = np.loadtxt(path, dtype=float)
series = MinMaxScaler(feature_range=(-1,1)).fit_transform(series.reshape(-1,1)).reshape(-1)
train_size = int(0.7*len(series))
num_part = 100
window_size = 25
train_series = series[:train_size]
test_series = series[train_size:]
partitions = partition_series(train_series, num_partitions=num_part)
partition_wins = []
partition_targets = []
partition_history = []
for label, p in enumerate(partitions):
    wins, targets = sliding_win_target(p, window_size, 1)
    partition_wins.append(wins)
    partition_targets.append(targets)
    partition_history.append(wins)
    
train_X, train_Y = sliding_win_target(train_series, window_size, 1)
dts = [DecisionTreeRegressor(max_depth=20) for p in range(num_part)]
#dts = [dts[p].fit(test_series,partition_targets[p]) for p in range(num_part)]
dts = [dts[p].fit(train_X, train_Y) for p in range(num_part)]

test_X, final_targets = sliding_win_target(test_series, window_size, 1)    
# preds = []
# for x in test_X:
#     tree_id = find_partition(partition_history, x)
#     preds.append(dts[tree_id].predict([x]))
# print(preds)
preds = [dts[p].predict(test_X) for p in range(num_part)]

# mse = mean_squared_error(y_true=final_targets, y_pred=preds)
# mape = mean_absolute_percentage_error(y_true=final_targets, y_pred=preds)
# print(f"MSE {mse} MAPE {mape}")
mins = []
for idx, p in enumerate(preds):
    mse = mean_squared_error(y_true=final_targets, y_pred=p)
    mape = mean_absolute_percentage_error(y_true=final_targets, y_pred=p)
    print(f"Id {idx} MSE {mse} MAPE {mape}")
    mins.append(mape)
best = np.min(mins)

train_X, train_Y = sliding_win_target(train_series, window_size, 1)
test_X, test_Y = sliding_win_target(test_series, window_size, 1)
rt = DecisionTreeRegressor(max_depth=20)
rt.fit(train_X, train_Y)
p_rt = rt.predict(test_X)

rt_mse = mean_squared_error(y_true=test_Y, y_pred=p_rt)
rt_mape = mean_absolute_percentage_error(y_true=test_Y, y_pred=p_rt)
print(f"RT MSE {rt_mse} MAPE {rt_mape}")

rf = RandomForestRegressor(n_estimators=50, max_depth=20, n_jobs=10)
rf.fit(train_X, train_Y)

p_rf = rf.predict(test_X)
rf_mse = mean_squared_error(y_true=test_Y, y_pred=p_rf)
rf_mape = mean_absolute_percentage_error(y_true=test_Y, y_pred=p_rf)
print(f"RT MSE {rf_mse} MAPE {rf_mape}")
print(f"BEst {best}")