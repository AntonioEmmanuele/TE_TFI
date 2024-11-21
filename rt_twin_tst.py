import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import os
from lib.ts_manip import sliding_win_target
ds = "datasets/processed"

def evaluate_series(train, validation, sizes):
    for ts in series:
        for ws in sizes:
            sliding_win_target(ts, ws, 1)
            model = DecisionTreeRegressor(max_depth=20)
            model.fit
def list_partitioning(a_list, num_of_partitions):
    return [list(item) for item in np.array_split(a_list, num_of_partitions)]

cpu_count = os.cpu_count()
list_series = os.listdir(ds)
time_series = []

for series in list_series:
    time_series.append(np.loadtxt(os.path.join(ds,series)))
ser_cpu = list_partitioning(time_series)
time_windows = [10, 20, 30, 40 , 50, 100, 150, 200, 500]
print(list_series)
