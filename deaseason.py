import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
from lib.ts_manip import sliding_win
# Load the time series data
file_path = './datasets/processed/232_UCR_Anomaly_mit14134longtermecg_8763_57530_57790.txt'
data = np.loadtxt(file_path, dtype=float)
# print(data)
# exit(1)

# Define sliding window size
window_size = 25

# Function to create sliding windows
def create_sliding_windows(series, window_size):
    return np.array([series[i:i+window_size] for i in range(len(series) - window_size + 1)])

# Create sliding windows for raw data
raw_windows = sliding_win(data, window_size)

# Preprocessing: Min-Max Scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Preprocessing: Detrending and Deseasonalization
# Using STL decomposition to remove seasonality and trend
result = seasonal_decompose(data_scaled, model='additive', period=window_size)
print(result)
# print(result.trend)
# print(result.seasonal)
# exit(1)
#data_detrended = data_scaled - result.trend
data_deseasonalized = data_scaled - result.seasonal
# print(data_deseasonalized)
# exit(1)
# Create sliding windows for preprocessed data
preprocessed_windows = sliding_win(data_deseasonalized, window_size)

# Perform clustering
def cluster_and_evaluate(windows, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(windows)
    score = silhouette_score(windows, clusters)
    return clusters, score

# Clustering raw and preprocessed data
raw_clusters, raw_score = cluster_and_evaluate(raw_windows)
preprocessed_clusters, preprocessed_score = cluster_and_evaluate(preprocessed_windows)

# Display the results
res = {
    "Silhouette Score (Raw)": raw_score,
    "Silhouette Score (Preprocessed)": preprocessed_score
}
print(res)