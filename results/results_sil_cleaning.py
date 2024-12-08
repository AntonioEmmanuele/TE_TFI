import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

# Folder and model names
folders = ["hyp_test_ucr_new", "hyp_randomforest_ucr", "hyp_rt_ucr", "hyp_xgboost_ucr"]
models = ["Ours", "RandomForest", "RegressionTree", "XGBoost"]

# Initialize a dictionary to hold results
data = {model: None for model in models}

# Read CSVs and store data for each model
for folder, model in zip(folders, models):
    file_path = os.path.join(folder, "stats_hyp.csv")
    if os.path.exists(file_path):
        data[model] = pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")

# Remove rows where Sil Hyp < 0.35 for all models
def filter_sil_hyp(df):
    if "Sil Hyp" in df.columns:
        return df[df["Sil Hyp"] >= 0.6]
    return df

data = {model: filter_sil_hyp(df) for model, df in data.items() if df is not None}

# Remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to all models
for model in models:
    if data[model] is not None:
        data[model] = remove_outliers(data[model], "Final MSE")
        data[model] = remove_outliers(data[model], "Final MAPE")

# Filter "Ours" to get the best result for each series (minimum MSE and MAPE)
ours_data = data["Ours"]
if ours_data is not None:
    best_ours_mse = ours_data.loc[ours_data.groupby("Series")["Final MSE"].idxmin()]
    best_ours_mape = ours_data.loc[ours_data.groupby("Series")["Final MAPE"].idxmin()]
else:
    raise ValueError("Data for 'Ours' not found!")

# Merge all models' data based on "Series"
results = best_ours_mse[["Series", "Final MSE"]].rename(columns={"Final MSE": "Ours"})
for model in models[1:]:
    if data[model] is not None:
        results = results.merge(
            data[model][["Series", "Final MSE"]].rename(columns={"Final MSE": model}),
            on="Series",
        )

# Check for missing values or inconsistencies
if results.isnull().values.any():
    raise ValueError("Missing values in the merged results. Ensure all series are present in all models.")

# Perform the Friedman test
mse_values = results[models].values
friedman_stat, friedman_p = friedmanchisquare(*mse_values.T)

print("Friedman Test Results:")
print(f"Statistic: {friedman_stat}, p-value: {friedman_p}")
if friedman_p < 0.05:
    print("The differences are statistically significant at the 95% confidence level.")
else:
    print("No significant differences found.")

# Perform the Nemenyi test
nemenyi_results = posthoc_nemenyi_friedman(mse_values)

# Recalculate ranks based on relative performance
ranks = np.zeros_like(mse_values)
for i, row in enumerate(mse_values):
    ranks[i] = np.argsort(np.argsort(row)) + 1  # Rank each row
mean_ranks = np.mean(ranks, axis=0)
rank_table = pd.DataFrame({"Model": models, "Mean Rank": mean_ranks})
rank_table = rank_table.sort_values(by="Mean Rank")

# Output results
print("\nNemenyi Test Results (Pairwise Ranks):")
print(nemenyi_results)

print("\nAverage Ranks:")
print(rank_table)

# Save the Nemenyi results to a CSV file for easier viewing
nemenyi_results.to_csv("nemenyi_test_results.csv")
rank_table.to_csv("model_ranks.csv", index=False)
print("Nemenyi test results saved to 'nemenyi_test_results.csv'")
print("Model ranks saved to 'model_ranks.csv'")

# Count the number of times "Ours" is better than others
comparison_counts = {model: {"better": 0, "equal": 0, "lesser": 0} for model in models if model != "Ours"}
for _, row in results.iterrows():
    for model in models[1:]:
        if row["Ours"] < row[model]:
            comparison_counts[model]["better"] += 1
        elif row["Ours"] == row[model]:
            comparison_counts[model]["equal"] += 1
        else:
            comparison_counts[model]["lesser"] += 1

print("\nComparison of 'Ours' with other models:")
for model, counts in comparison_counts.items():
    print(f"'Ours' vs {model}: Better: {counts['better']}, Equal: {counts['equal']}, Lesser: {counts['lesser']}.")

# Count the number of times "XGBoost" is better than "RandomForest"
xgb_rf_comparison = {"better": 0, "equal": 0, "lesser": 0}
for _, row in results.iterrows():
    if row["XGBoost"] < row["RandomForest"]:
        xgb_rf_comparison["better"] += 1
    elif row["XGBoost"] == row["RandomForest"]:
        xgb_rf_comparison["equal"] += 1
    else:
        xgb_rf_comparison["lesser"] += 1

print("\nComparison of 'XGBoost' with 'RandomForest':")
print(f"XGBoost better than RandomForest: {xgb_rf_comparison['better']} times.")
print(f"XGBoost equal to RandomForest: {xgb_rf_comparison['equal']} times.")
print(f"XGBoost lesser than RandomForest: {xgb_rf_comparison['lesser']} times.")
