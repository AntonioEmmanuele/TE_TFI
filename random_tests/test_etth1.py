import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from lib.ts_manip import sliding_win_cluster_aware
#from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from src.te_tfi_model import TE_TFI
from scipy.stats import anderson
from sklearn.datasets import make_blobs
from tslearn.clustering import KShape
import pandas as pd

def create_sliding_windows(data, window_size, target_column, step=1):
    """
    Crea finestre sovrapposte (sliding windows) da una colonna del dataset.

    Args:
        data (pd.DataFrame): Dataset originale.
        window_size (int): Lunghezza della finestra.
        target_column (str): Nome della colonna su cui applicare le sliding windows.
        step (int): Dimensione dello step tra le finestre (default=1).
    
    Returns:
        np.ndarray: Matrice con le sliding windows.
    """
    series = data[target_column].values
    num_windows = (len(series) - window_size) // step + 1
    return np.array([
        series[i : i + window_size] 
        for i in range(0, num_windows * step, step)
    ])

def create_sliding_windows_multifeature_with_target(data, window_size, target_column, step=1):
    """
    Crea sliding windows per le feature e il target.

    Args:
        data (pd.DataFrame): Dataset originale.
        window_size (int): Lunghezza della finestra.
        target_column (str): Nome della colonna target.
        step (int): Dimensione dello step tra le finestre (default=1).
    
    Returns:
        np.ndarray: Sliding windows delle feature (n_windows, window_size, n_features).
        np.ndarray: Valori target corrispondenti (n_windows,).
    """
    num_samples = len(data)
    num_windows = (num_samples - window_size) // step + 1

    # Separare le feature dal target
    feature_columns = data.drop(columns=[target_column]).values
    target_series = data[target_column].values

    # Creazione sliding windows per le feature
    feature_windows = np.array([
        feature_columns[i : i + window_size]
        for i in range(0, num_windows * step, step)
    ])

    # Creazione del target associato a ogni finestra
    targets = np.array([
        target_series[i + window_size - 1]  # Target corrisponde all'ultimo elemento della finestra
        for i in range(0, num_windows * step, step)
    ])

    return feature_windows, targets

# Esempio d'uso
window_size = 24  # Finestra di 24 ore
step = 1
target_column = 'OT'  # Colonna di interesse

df = pd.read_csv("./datasets/etth/ETTh1.csv")
sliding_windows, target = create_sliding_windows_multifeature_with_target(df, window_size, target_column, step)
print(f"Shape of sliding windows: {sliding_windows.shape}")
print("Example sliding window:", sliding_windows[0])
print("Target,", target[0] )
print(df.iloc[23])
