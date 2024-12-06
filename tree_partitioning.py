from sklearn.tree import DecisionTreeClassifier
from lib.ts_manip import partition_series, sliding_win, compute_l2_norm
import numpy as np

def compute_av_l2(set_of_samples, single_sample):
    values = []
    for s in set_of_samples:
        values.append(compute_l2_norm(s,single_sample))
    return np.mean(values)

path = "./datasets/processed/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt"
series = np.loadtxt(path, dtype=float)
train_size = int(0.7*len(series))
num_part = 5
window_size = 25
train_series = series[:train_size]
test_series = series[train_size:]
partitions = partition_series(train_series, num_partitions=num_part)
sliding_wins = []
partition_history = []
labels= []
for label, p in enumerate(partitions):
    wins = sliding_win(p, window_size)
    partition_history.append(wins)
    sliding_wins.extend(wins)
    labels.extend([label for i in wins])

dt = DecisionTreeClassifier(max_depth=20, criterion='squared_error')
dt.fit(sliding_wins, labels)
test_samples = sliding_win(test_series)
test_labels = dt.predict(test_samples)

# Now evaluate the mean error using the L2 norm.
correctly = 0
non_correctly = 0
for test_idx in range(len(test_labels)):
    norms = []
    for i in range(0,num_part):
        norms.append(compute_av_l2(partition_history[i], test_samples[test_idx]))
    correct_label = np.argmin(norms)
    if correct_label == test_labels[test_idx]:
        correctly +=1
    else:
        non_correctly += 1

# Finally print acc


