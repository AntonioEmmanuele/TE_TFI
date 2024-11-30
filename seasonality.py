import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.fftpack import fft
 
# Directory containing UCR time series files
ucr_directory = "./datasets/processed/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt"
 
# Function to compute seasonality using autocorrelation
def detect_seasonality_autocorrelation(time_series, max_lag=100):
    autocorr = acf(time_series, nlags=max_lag, fft=True)
    significant_lags = np.where(autocorr > 0.3)[0]  # Adjust threshold if needed
    return significant_lags
 
# Function to compute seasonality using Fourier Transform
def detect_seasonality_fourier(time_series):
    freq_amplitude = np.abs(fft(time_series))
    dominant_frequencies = np.where(freq_amplitude > np.mean(freq_amplitude))[0]
    return dominant_frequencies
 
# Analyze each file in the UCR directory
seasonality_results = []
 
for file_name in os.listdir(ucr_directory):
    if file_name.endswith(".csv"):  # Assuming time series are in CSV files
        file_path = os.path.join(ucr_directory, file_name)
        time_series = pd.read_csv(file_path, header=None).iloc[:, 0].values
 
        # Autocorrelation Analysis
        significant_lags = detect_seasonality_autocorrelation(time_series)
 
        # Fourier Transform Analysis
        dominant_frequencies = detect_seasonality_fourier(time_series)
 
        # Record results
        has_seasonality = len(significant_lags) > 0 or len(dominant_frequencies) > 0
        seasonality_results.append({
            "file": file_name,
            "has_seasonality": has_seasonality,
            "significant_lags": significant_lags.tolist(),
            "dominant_frequencies": dominant_frequencies.tolist()
        })
 
# Save results to a CSV file
results_df = pd.DataFrame(seasonality_results)
results_df.to_csv("ucr_seasonality_results.csv", index=False)
 
print("Seasonality analysis complete. Results saved to 'ucr_seasonality_results.csv'.")