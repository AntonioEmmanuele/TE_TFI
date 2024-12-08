import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

# Path to "Ours" folder and file
folder = "hyp_test_ucr_new"
file_path = os.path.join(folder, "stats_hyp.csv")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the data
ours_data = pd.read_csv(file_path)

# Filter out rows with "Sil Hyp" < 0.35
ours_data = ours_data[ours_data["Sil Hyp"] >= 0.35]

# Convert "Trees MSE Hyp" and "Trees MAPE Hyp" from strings to numpy arrays

def parse_np_array(column):
    def safe_eval(val):
        try:
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                return np.array([float(x.split("(")[1].strip(")")) for x in val.strip("[]").split(",")])
            return np.array([])
        except Exception:
            return np.array([])

    return column.apply(lambda x: safe_eval(x) if isinstance(x, str) else np.array([]))

ours_data["Trees MSE Hyp"] = parse_np_array(ours_data["Trees MSE Hyp"])
ours_data["Trees MAPE Hyp"] = parse_np_array(ours_data["Trees MAPE Hyp"])

# Calculate the mean Trees MSE Hyp for each series
ours_data["Mean Trees MSE Hyp"] = ours_data["Trees MSE Hyp"].apply(lambda x: np.mean(x) if x.size > 0 else np.nan)

# Drop rows with NaN values in "Mean Trees MSE Hyp"
ours_data = ours_data.dropna(subset=["Mean Trees MSE Hyp"])

# Extract Silhouette scores and Mean Trees MSE
sil_scores = ours_data["Sil Hyp"]
mean_trees_mse = ours_data["Mean Trees MSE Hyp"]

# Calculate correlation
correlation, p_value = spearmanr(sil_scores, mean_trees_mse)

print("Correlation Results:")
print(f"Spearman correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

# Interpretation of the p-value
if p_value < 0.05:
    print("The correlation is statistically significant at the 5% significance level.")
else:
    print("The correlation is not statistically significant at the 5% significance level.")

# Scatter plot for data points
plt.figure(figsize=(8, 6))
plt.scatter(sil_scores, mean_trees_mse, alpha=0.7, label="Data Points")
plt.title("Scatter Plot: Silhouette Scores vs. Mean Trees MSE")
plt.xlabel("Silhouette Score")
plt.ylabel("Mean Trees MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_vs_mse_scatter.png")
plt.show()

print("Scatter plot saved as 'silhouette_vs_mse_scatter.png'")

# Separate plot for the trend line
plt.figure(figsize=(8, 6))
plt.title("Trend Line: Silhouette Scores vs. Mean Trees MSE")
plt.xlabel("Silhouette Score")
plt.ylabel("Mean Trees MSE")
plt.grid(True)

# Define functions for trend fitting
# Linear function
def linear(x, a, b):
    return a * x + b

# Quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Exponential function
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit and plot the best trend
try:
    linear_params, _ = curve_fit(linear, sil_scores, mean_trees_mse)
    quadratic_params, _ = curve_fit(quadratic, sil_scores, mean_trees_mse)
    exponential_params, _ = curve_fit(exponential, sil_scores, mean_trees_mse, maxfev=10000)

    # Generate x values for trend lines
    trend_x = np.linspace(sil_scores.min(), sil_scores.max(), 500)

    # Calculate trend lines
    linear_y = linear(trend_x, *linear_params)
    quadratic_y = quadratic(trend_x, *quadratic_params)
    exponential_y = exponential(trend_x, *exponential_params)

    # Plot each trend line
    plt.plot(trend_x, linear_y, label="Linear Trend", color="red")
    plt.plot(trend_x, quadratic_y, label="Quadratic Trend", linestyle="--", color="blue")
    plt.plot(trend_x, exponential_y, label="Exponential Trend", linestyle="-.", color="green")
except Exception as e:
    print(f"Error fitting trend lines: {e}")

plt.legend()
plt.tight_layout()
plt.savefig("silhouette_vs_mse_trend.png")
plt.show()

print("Trend line plot saved as 'silhouette_vs_mse_trend.png'")
