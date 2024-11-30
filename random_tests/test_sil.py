import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from sklearn.metrics import silhouette_score

# Load a sample time series dataset (ECG dataset for example)
x_train, y_train, x_test, y_test = CachedDatasets().load_dataset("Trace")

# Use only the train set for clustering
X = x_train

# Set up the parameters for the KMeans algorithm
n_clusters = 3  # you can choose a different value based on your data

# Apply KMeans
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=0)
y_pred = model.fit_predict(X)
print(y_pred)

# Calculate the silhouette score to evaluate the clustering
# Reshape data to 2D array for silhouette score calculation
X_2d = X.reshape((X.shape[0], -1))
silhouette_avg = silhouette_score(X_2d, y_pred)

print(f"The average silhouette score for {n_clusters} clusters is: {silhouette_avg:.2f}")
