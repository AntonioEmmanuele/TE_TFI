import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tslearn.clustering import KShape
from sklearn.cluster import AgglomerativeClustering
from lib.ts_manip import partition_series, partition_series_multivariate
from sklearn.cluster import DBSCAN
from tslearn.metrics import cdist_dtw
from statsmodels.tsa.stattools import adfuller

# Load the series from the provided file#
#file_path = './datasets/processed/024_UCR_Anomaly_DISTORTEDInternalBleeding10_3200_4526_4556.txt'
#series = np.loadtxt(file_path)
file_path = "./datasets/etth/ETTH1.csv"
series = pd.read_csv(file_path)
series['date'] = pd.to_datetime(series['date'])
series['date'] = (series['date'] - series['date'].min()).dt.total_seconds()

def make_stationary(data, method='difference', lag=1):
    """
    Makes a multivariate time series stationary by differencing or log transformation.
    
    Parameters:
        data (pd.DataFrame): The multivariate time series as a Pandas DataFrame.
        method (str): Method to make the series stationary ('difference' or 'log_difference').
        lag (int): Lag to use for differencing (default=1).
        
    Returns:
        pd.DataFrame: A DataFrame of the stationary time series.
        dict: ADF test results for each column.
    """
    stationary_data = data.copy()
    adf_results = {}
    
    for column in stationary_data.columns:
        if method == 'difference':
            # Apply differencing
            stationary_data[column] = stationary_data[column].diff(periods=lag).dropna()
        elif method == 'log_difference':
            # Apply log transformation and then differencing
            stationary_data[column] = np.log(stationary_data[column]).diff(periods=lag).dropna()
        else:
            raise ValueError("Method must be 'difference' or 'log_difference'.")
        
        # Drop NaN values introduced by differencing
        stationary_data[column] = stationary_data[column].dropna()
        
        # Perform ADF test to check stationarity
        adf_result = adfuller(stationary_data[column].dropna())
        adf_results[column] = {
            "ADF Statistic": adf_result[0],
            "p-value": adf_result[1],
            "Critical Values": adf_result[4],
            "Stationary": adf_result[1] < 0.05  # p-value < 0.05 indicates stationarity
        }
    
    return stationary_data, adf_results

def z_normalize_multivariate(df, exclude_columns=None):
    """
    Z-normalizes each column of a multivariate time series from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        exclude_columns (list, optional): List of column names to exclude from normalization 
                                           (e.g., time column, non-numeric columns).
    
    Returns:
        pd.DataFrame: A DataFrame with Z-normalized columns.
    """

    # Exclude specific columns if needed
    if exclude_columns is not None:
        exclude_columns = exclude_columns if isinstance(exclude_columns, list) else [exclude_columns]
        columns_to_normalize = df.columns.difference(exclude_columns)
    else:
        columns_to_normalize = df.columns
    
    # Apply Z-normalization to each column
    for col in columns_to_normalize:
        if pd.api.types.is_numeric_dtype(df[col]):  # Normalize only numeric columns
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    
    return df

def sliding_window_multivariate(data, window_size, target_column, step=1):
    """
    Applica una sliding window a una serie temporale multivariata e genera target da una colonna specifica.

    Parameters:
        data (pd.DataFrame): DataFrame con i dati multivariati.
        window_size (int): Dimensione della finestra scorrevole.
        target_column (str): Nome della colonna da utilizzare come target.
        step (int): Passo della finestra (default=1).
    
    Returns:
        X (np.ndarray): Array con le finestre sulle feature.
        y (np.ndarray): Array con i valori target associati.
    """
    X, y = [], []
    # print(data.iloc[:2])
    # Itera sulle righe usando la dimensione della finestra
    for i in range(0, len(data) - window_size, step):
        # Estrai una finestra (tutte le colonne tranne il target)
        window = data.iloc[i:i + 2, :].values.reshape(-1)
       #window.reshape(-1)
        # print(window)
        # exit(1)
        # Aggiungi la finestra alle feature
        X.append(window)
        
        # Estrai il valore target corrispondente (successivo alla finestra)
        target_index = i + window_size
        y.append(data.iloc[target_index][target_column])
    
    return np.array(X), np.array(y)

# Z-normalization
def z_normalize(values):
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / std

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

# series = z_normalize(series)
series = z_normalize_multivariate(series, ["OT"])
#sliding_wins, targets = sliding_window_multivariate(series, 24, "OT", 1)
#dtw = cdist_dtw(sliding_wins, n_jobs= 8, verbose=True)
sliding_wins = partition_series_multivariate(series, 24)

results = []

# Direct clustering on raw series
kmeans_raw = KMeans(n_clusters = 10, max_iter = 10000, random_state = 42)
clusters_raw = kmeans_raw.fit_predict(sliding_wins)

# Calculate silhouette score for raw approach
silhouette_raw = silhouette_score(sliding_wins, clusters_raw)
results.append({'Approach': 'Raw Series', 'Silhouette Score': silhouette_raw})
print(results)
exit(1)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(sliding_wins)

silhouette_dbscan = silhouette_score(sliding_wins, dbscan_labels)
results.append({'Approach': 'DB SCAN', 'Silhouette Score': silhouette_dbscan})
print("Running agglomerative")
agg_cluster = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="average", verbose = True)
hier_res = agg_cluster.fit_predict(dtw)
silhouette_avg = silhouette_score(dtw, hier_res, metric="precomputed")
results.append(silhouette_avg)
# # Clustering con K-Shape
# kshape = KShape(n_clusters = 3, random_state=42)
# clusters_kshape = kshape.fit_predict(reshaped_series)

# # Calcolo del silhouette score per K-Shape
# silhouette_kshape = silhouette_score(reshaped_series, clusters_kshape)
print(results)
#print(silhouette_kshape)
# # Convert results to DataFrame and display
# results_df = pd.DataFrame(results)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Clustering Approach Comparison", dataframe=results_df)
