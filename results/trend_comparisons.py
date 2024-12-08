import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit

# Path to "Ours" folder and file
folder = "hyp_test_ucr_new"
file_path = os.path.join(folder, "stats_hyp.csv")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the data
ours_data = pd.read_csv(file_path)

# Convert "Trees MSE Hyp" from strings to numpy arrays
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

# Calculate the mean Trees MSE Hyp for each series
ours_data["Mean Trees MSE Hyp"] = ours_data["Trees MSE Hyp"].apply(lambda x: np.mean(x) if x.size > 0 else np.nan)

# Normalize MSE values to the range [0, 1]
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

ours_data["Normalized Mean Trees MSE Hyp"] = normalize(ours_data["Mean Trees MSE Hyp"])

# Remove outliers using the mean + std technique
def remove_outliers(series):
    mean_val = series.mean()
    std_dev = series.std()
    return series[(series >= mean_val - std_dev) & (series <= mean_val + std_dev)]

ours_data = ours_data[ours_data["Normalized Mean Trees MSE Hyp"].notna()]
ours_data["Normalized Mean Trees MSE Hyp"] = remove_outliers(ours_data["Normalized Mean Trees MSE Hyp"])

# Drop rows with NaN values after outlier removal
ours_data = ours_data.dropna(subset=["Normalized Mean Trees MSE Hyp"])

# Extract Silhouette scores and Normalized Mean Trees MSE
sil_scores = ours_data["Sil Hyp"]
normalized_mean_trees_mse = ours_data["Normalized Mean Trees MSE Hyp"]

# Calculate Spearman correlation
spearman_corr, spearman_p_value = spearmanr(sil_scores, normalized_mean_trees_mse)
print("Spearman Correlation Results:")
print(f"Spearman correlation coefficient: {spearman_corr}")
print(f"P-value: {spearman_p_value}")

# Calculate Pearson correlation
pearson_corr, pearson_p_value = pearsonr(sil_scores, normalized_mean_trees_mse)
print("Pearson Correlation Results:")
print(f"Pearson correlation coefficient: {pearson_corr}")
print(f"P-value: {pearson_p_value}")

# Interpretation of the p-values
if spearman_p_value < 0.05:
    print("The Spearman correlation is statistically significant at the 5% significance level.")
else:
    print("The Spearman correlation is not statistically significant at the 5% significance level.")

if pearson_p_value < 0.05:
    print("The Pearson correlation is statistically significant at the 5% significance level.")
else:
    print("The Pearson correlation is not statistically significant at the 5% significance level.")

# Fit models for linear, quadratic, and exponential trends
# Linear function
def linear(x, a, b):
    return a * x + b

# Quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Exponential function
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit each model
linear_params, _ = curve_fit(linear, sil_scores, normalized_mean_trees_mse)
quadratic_params, _ = curve_fit(quadratic, sil_scores, normalized_mean_trees_mse)
exponential_params, _ = curve_fit(exponential, sil_scores, normalized_mean_trees_mse, maxfev=10000)

# Generate x values for trend lines
trend_x = np.linspace(sil_scores.min(), sil_scores.max(), 500)

# Calculate trend lines
linear_y = linear(trend_x, *linear_params)
quadratic_y = quadratic(trend_x, *quadratic_params)
exponential_y = exponential(trend_x, *exponential_params)

# Save residual sums of squares (RSS)
linear_residuals = normalized_mean_trees_mse - linear(sil_scores, *linear_params)
quadratic_residuals = normalized_mean_trees_mse - quadratic(sil_scores, *quadratic_params)
exponential_residuals = normalized_mean_trees_mse - exponential(sil_scores, *exponential_params)

rss = {
    "Linear": np.sum(linear_residuals**2),
    "Quadratic": np.sum(quadratic_residuals**2),
    "Exponential": np.sum(exponential_residuals**2)
}

best_fit = min(rss, key=rss.get)

print("Residual Sum of Squares (RSS):")
for model, value in rss.items():
    print(f"{model}: {value}")
print(f"Best fit based on RSS: {best_fit}")

# Scatter plot for data points
plt.figure(figsize=(8, 6))
plt.scatter(sil_scores, normalized_mean_trees_mse, alpha=0.7, label="Data Points")
plt.title("Scatter Plot: Silhouette Scores vs. Normalized Mean Trees MSE")
plt.xlabel("Silhouette Score")
plt.ylabel("Normalized Mean Trees MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_vs_mse_scatter.png")
plt.show()

print("Scatter plot saved as 'silhouette_vs_mse_scatter.png'")

# Separate plot for trends
plt.figure(figsize=(8, 6))
plt.plot(trend_x, linear_y, label="Linear Trend", color="red")
plt.plot(trend_x, quadratic_y, label="Quadratic Trend", linestyle="--", color="blue")
plt.plot(trend_x, exponential_y, label="Exponential Trend", linestyle="-.", color="green")
plt.title("Trend Lines: Silhouette Scores vs. Normalized Mean Trees MSE")
plt.xlabel("Silhouette Score")
plt.ylabel("Normalized Mean Trees MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_vs_mse_trends.png")
plt.show()

print("Trend line plot saved as 'silhouette_vs_mse_trends.png'")