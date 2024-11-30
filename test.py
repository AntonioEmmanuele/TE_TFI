import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tslearn.clustering import KShape
from lib.ts_manip import sliding_win
# Load the series from the provided file
file_path = './datasets/processed/024_UCR_Anomaly_DISTORTEDInternalBleeding10_3200_4526_4556.txt'
series = np.loadtxt(file_path)

# Create a DataFrame to store results for comparison
results = []

# Feature extraction approach
def extract_features(series):
    return {
        'mean': np.mean(series),
        'std_dev': np.std(series),
        'skewness': skew(series),
        'kurtosis': kurtosis(series),
        'max_value': np.max(series),
        'min_value': np.min(series),
        'dominant_frequency': np.argmax(np.abs(fft(series)[1:])) + 1,  # Dominant frequency (ignoring DC)
        'autocorr_lag1': np.corrcoef(series[:-1], series[1:])[0, 1],
    }

# Create windows for feature extraction
window_size = 300
step_size = 100
features_list = []

# for start in range(0, len(series) - window_size, step_size):
#     window = series[start:start + window_size]
#     features_list.append(extract_features(window))
sliding_wins = sliding_win(series, 100)
for win in sliding_wins:
    features_list.append(extract_features(win))     
features_df = pd.DataFrame(features_list)

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Clustering on extracted features
kmeans_features = KMeans(n_clusters=3, random_state=42)
clusters_features = kmeans_features.fit_predict(scaled_features)

# Calculate silhouette score for features approach
silhouette_features = silhouette_score(scaled_features, clusters_features)
results.append({'Approach': 'Feature Extraction', 'Silhouette Score': silhouette_features})

# Direct clustering on raw series
#reshaped_series = series[:len(series) // window_size * window_size].reshape(-1, window_size)  # Reshape for clustering
kmeans_raw = KMeans(n_clusters=3, random_state=42)
clusters_raw = kmeans_raw.fit_predict(sliding_wins)

# Calculate silhouette score for raw approach
silhouette_raw = silhouette_score(sliding_wins, clusters_raw)
results.append({'Approach': 'Raw Series', 'Silhouette Score': silhouette_raw})
print(results)
# # Clustering con K-Shape
# kshape = KShape(n_clusters = 3, random_state=42)
# clusters_kshape = kshape.fit_predict(reshaped_series)

# # Calcolo del silhouette score per K-Shape
# silhouette_kshape = silhouette_score(reshaped_series, clusters_kshape)

# print(results)
# print(silhouette_kshape)
# # # Convert results to DataFrame and display
# # results_df = pd.DataFrame(results)
# # import ace_tools as tools; tools.display_dataframe_to_user(name="Clustering Approach Comparison", dataframe=results_df)
