# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import friedmanchisquare
# from scikit_posthocs import posthoc_nemenyi_friedman
# from matplotlib.backends.backend_pdf import PdfPages

# # Folder and model names
# folders = ["../experiments/ucr_no_preprocess/hyp_test_ucr_new", 
#            "../experiments/ucr_no_preprocess/hyp_randomforest_ucr", 
#            "../experiments/ucr_no_preprocess/hyp_rt_ucr", 
#            "../experiments/ucr_no_preprocess/hyp_xgboost_ucr"]
# models = ["Ours", "RandomForest", "RegressionTree", "XGBoost"]

# # Initialize a dictionary to hold results
# data = {model: None for model in models}

# # Read CSVs and store data for each model
# for folder, model in zip(folders, models):
#     file_path = os.path.join(folder, "stats_hyp.csv")
#     if os.path.exists(file_path):
#         data[model] = pd.read_csv(file_path)
#     else:
#         print(f"File not found: {file_path}")

# # Remove rows where Sil Hyp < 0.35 for all models
# def filter_sil_hyp(df):
#     # if "Sil Hyp" in df.columns:
#     #     return df[df["Sil Hyp"] >= 0.3]
#     return df

# data = {model: filter_sil_hyp(df) for model, df in data.items() if df is not None}

# # Filter "Ours" to get the best result for each series (minimum MSE and MAPE)
# ours_data = data["Ours"]
# if ours_data is not None:
#     best_ours_mse = ours_data.loc[ours_data.groupby("Series")["Final MSE"].idxmin()]
#     best_ours_mape = ours_data.loc[ours_data.groupby("Series")["Final MSE"].idxmin()]
# else:
#     raise ValueError("Data for 'Ours' not found!")

# # Merge all models' data based on "Series"
# results = best_ours_mse[["Series", "Final MSE"]].rename(columns={"Final MSE": "Ours"})
# for model in models[1:]:
#     if data[model] is not None:
#         results = results.merge(
#             data[model][["Series", "Final MSE"]].rename(columns={"Final MSE": model}),
#             on="Series",
#         )

# # Check for missing values or inconsistencies
# if results.isnull().values.any():
#     raise ValueError("Missing values in the merged results. Ensure all series are present in all models.")

# # Perform the Friedman test before outlier removal
# mse_values = results[models].values
# friedman_stat, friedman_p = friedmanchisquare(*mse_values.T)

# print("Friedman Test Results:")
# print(f"Statistic: {friedman_stat}, p-value: {friedman_p}")
# if friedman_p < 0.05:
#     print("The differences are statistically significant at the 95% confidence level.")
# else:
#     print("No significant differences found.")

# # Perform the Nemenyi test before outlier removal
# nemenyi_results = posthoc_nemenyi_friedman(mse_values)

# # Recalculate ranks based on relative performance
# ranks = np.zeros_like(mse_values)
# for i, row in enumerate(mse_values):
#     ranks[i] = np.argsort(np.argsort(row)) + 1  # Rank each row
# mean_ranks = np.mean(ranks, axis=0)
# rank_table = pd.DataFrame({"Model": models, "Mean Rank": mean_ranks})
# rank_table = rank_table.sort_values(by="Mean Rank")

# # Output results
# print("\nNemenyi Test Results (Pairwise Ranks):")
# print(nemenyi_results)

# print("\nAverage Ranks:")
# print(rank_table)

# # Save the Nemenyi results to a CSV file for easier viewing
# nemenyi_results.to_csv("nemenyi_test_results.csv")
# rank_table.to_csv("model_ranks.csv", index=False)
# print("Nemenyi test results saved to 'nemenyi_test_results.csv'")
# print("Model ranks saved to 'model_ranks.csv'")

# # Count the number of times "Ours" is better than others
# comparison_counts = {model: {"better": 0, "equal": 0, "lesser": 0} for model in models if model != "Ours"}
# for _, row in results.iterrows():
#     for model in models[1:]:
#         if row["Ours"] < row[model]:
#             comparison_counts[model]["better"] += 1
#         elif row["Ours"] == row[model]:
#             comparison_counts[model]["equal"] += 1
#         else:
#             comparison_counts[model]["lesser"] += 1

# print("\nComparison of 'Ours' with other models:")
# for model, counts in comparison_counts.items():
#     print(f"'Ours' vs {model}: Better: {counts['better']}, Equal: {counts['equal']}, Lesser: {counts['lesser']}.")

# # Count the number of times "XGBoost" is better than "RandomForest"
# xgb_rf_comparison = {"better": 0, "equal": 0, "lesser": 0}
# for _, row in results.iterrows():
#     if row["XGBoost"] < row["RandomForest"]:
#         xgb_rf_comparison["better"] += 1
#     elif row["XGBoost"] == row["RandomForest"]:
#         xgb_rf_comparison["equal"] += 1
#     else:
#         xgb_rf_comparison["lesser"] += 1

# print("\nComparison of 'XGBoost' with 'RandomForest':")
# print(f"XGBoost better than RandomForest: {xgb_rf_comparison['better']} times.")
# print(f"XGBoost equal to RandomForest: {xgb_rf_comparison['equal']} times.")
# print(f"XGBoost lesser than RandomForest: {xgb_rf_comparison['lesser']} times.")

# # --- Cleaning Anomalous MAPE and Plotting ---
# # Apply outlier removal to all models (after tests)
# def remove_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# for model in models:
#     if data[model] is not None:
#         data[model] = remove_outliers(data[model], "Final MSE")
#         #data[model] = remove_outliers(data[model], "Final MAPE")

# # # Clean MAPE > 0 and calculate mean/std
# # for model in models:
# #     if data[model] is not None:
# #         data[model] = data[model][data[model]["Final MAPE"] <= 1.0]
# #         mean_mape = data[model]["Final MAPE"].mean()
# #         std_mape = data[model]["Final MAPE"].std()
# #         print(f"{model} Mean MAPE: {mean_mape}, Std MAPE: {std_mape}")

# # Plot and save to PDF
# with PdfPages("model_comparison.pdf") as pdf:
#     for model in models:
#         if data[model] is not None:
#             plt.figure(figsize=(8, 6))
#             plt.boxplot(data[model]["Final MSE"])
#             plt.title(f"Boxplot of Final MSE - {model}")
#             plt.ylabel("MSE")
#             pdf.savefig()
#             plt.close()

#     print("Graphs saved to 'model_comparison.pdf'.")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman, sign_array, critical_difference_diagram
from matplotlib.backends.backend_pdf import PdfPages
def filter_by_ours_series(models_dict, ours_model):
    """
    Filtra i dataframe nel dizionario per mantenere solo le righe dove la colonna 'Series'
    è presente nel modello 'Ours'.
    
    :param models_dict: Dizionario contenente i dataframe per ogni modello.
    :param ours_model: Lista o array di 'Series' che appartengono al modello Ours.
    :return: Nuovo dizionario con i dataframe filtrati.
    """
    filtered_models = {}
    
    for model_name, df in models_dict.items():
        # Filtra il dataframe dove la colonna 'Series' è presente nel modello Ours
        filtered_df = df[df['Series'].isin(ours_model["Series"])]
        filtered_models.update({ model_name : filtered_df})
    return filtered_models

def sort_by_median(mape_data, models):
    # Calcola la mediana per ogni array di mape_data
    median_values = [np.median(mape) for mape in mape_data]
    
    # Ottieni gli indici che ordinano le mediane in ordine decrescente
    sorted_indices = np.argsort(median_values)[::-1]
    
    # Ordina sia mape_data che models in base agli indici ordinati
    sorted_mape_data = [mape_data[i] for i in sorted_indices]
    sorted_models = [models[i] for i in sorted_indices]
    
    return sorted_mape_data, sorted_models

# Folder and model names
folders = [
            "../experiments/ucr_no_preprocess/hyp_test_ucr_new",
            "../experiments/ucr_no_preprocess/hyp_rt_ucr", 
            "../experiments/ucr_no_preprocess/hyp_randomforest_ucr", 
           "../experiments/ucr_no_preprocess/hyp_xgboost_ucr"]
models = [ "Ours",  "RegressionTree", "RandomForest", "XGBoost" ]

# Initialize a dictionary to hold results
data = {model: None for model in models}


# def get_min_mse_per_series(csv_path):
#     # Read the CSV into a DataFrame
#     df = pd.read_csv(csv_path)
    
#     # Group by 'Series' and select the row with the minimum 'Final MSE' per group
#     # The idxmin() function returns the index of the minimum value in that column.
#     df_min = df.loc[df.groupby('Series')['Final MSE'].idxmin()]

#     return df_min

# Read CSVs and store data for each model
for folder, model in zip(folders, models):
    file_path = os.path.join(folder, "stats_hyp.csv")
    if os.path.exists(file_path):
        data[model] = pd.read_csv(file_path)     
    else:
        print(f"File not found: {file_path}")

# Filter "Ours" to get the best result for each series (minimum MSE and MAPE)
# print(data["Ours"])
# exit(1)
ours_data = data["Ours"]
if ours_data is not None:
    series = ours_data.loc[ours_data.groupby("Series")["Final MSE"].idxmin()]
    best_ours_mse = ours_data.loc[ours_data.groupby("Series")["Final MSE"].idxmin()]
else:
    raise ValueError("Data for 'Ours' not found!")
data["Ours"] = best_ours_mse
data = filter_by_ours_series(data, data["Ours"] )
#Merge all models' data based on "Series"
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

# Perform the Friedman test before outlier removal (using MSE)
mse_values = results[models].values
friedman_stat, friedman_p = friedmanchisquare(*mse_values.T)

print("Friedman Test Results:")
print(f"Statistic: {friedman_stat}, p-value: {friedman_p}")
if friedman_p < 0.05:
    print("The differences are statistically significant at the 95% confidence level.")
else:
    print("No significant differences found.")

# Perform the Nemenyi test before outlier removal (using MSE)
nemenyi_results = posthoc_nemenyi_friedman(mse_values)

# Recalculate ranks based on relative performance
ranks = np.zeros_like(mse_values)
for i, row in enumerate(mse_values):
    ranks[i] = np.argsort(np.argsort(row)) + 1  # Rank each row
mean_ranks = np.mean(ranks, axis=0)
rank_table = pd.DataFrame({"Model": models, "Mean Rank": mean_ranks})
#rank_table = pd.Series({"Model": models, "Mean Rank": mean_ranks})
#rank_table = pd.DataFrame([{m :r } for m,r in zip(models, mean_ranks)])

# print(rank_table)
rank_table = rank_table.sort_values(by="Mean Rank")

# Output results
print("\nNemenyi Test Results (Pairwise Ranks):")
print(nemenyi_results)

print("\nAverage Ranks:")
print(rank_table)
rank_table = rank_table.reset_index(drop=True)

#sign_plot(nemenyi_results)
plt.figure(figsize=(10, 2), dpi=1600)
# print(rank_table)
# exit(1)
print(nemenyi_results.values)
print(sign_array(nemenyi_results, alpha=0.05))
# exit(1)
critical_difference_diagram(ranks = rank_table["Mean Rank"], sig_matrix=sign_array(nemenyi_results))
plt.savefig("color_plot.pdf", format = "pdf", dpi = 1600)
# Save the Nemenyi results to a CSV file for easier viewing
nemenyi_results.to_csv("nemenyi_test_results.csv")
rank_table.to_csv("model_ranks.csv", index=False)
print("Nemenyi test results saved to 'nemenyi_test_results.csv'")
print("Model ranks saved to 'model_ranks.csv'")

# --- Cleaning Anomalous MAPE and Plotting ---
# Apply outlier removal to all models (after tests)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 -  IQR
    upper_bound = Q3 +  4 * IQR
    #return df[(df[column] >=  lower_bound) & (df[column] <=  upper_bound)]
    return df[(df[column] <= upper_bound)]


# Clean MAPE > 0 and calculate mean/std
for model in models:
    if data[model] is not None:
        data[model] = data[model][data[model]["Final MAPE"] <= 100]

for model in models:
    if data[model] is not None:
        #data[model] = remove_outliers(data[model], "Final MSE")
        data[model] = remove_outliers(data[model], "Final MAPE")

# Plot and save all MAPE boxplots to a single PDF
with PdfPages("mape_comparison.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting (MAPE for each model)
    mape_data = [data[model]["Final MAPE"] for model in models if data[model] is not None]
    mape_data, models = sort_by_median(mape_data, models)
    # Create boxplot
    plt.boxplot(mape_data, labels=models)
    #plt.title("Boxplot of Final MAPE for Different Models")
    plt.xlabel("MAPE", fontsize = 14)
    plt.xticks(fontsize=14)  # Set the font size of x-ticks
    plt.yticks(fontsize=14)  # Set the font size of x-ticks

    pdf.savefig()  # Save the figure
    plt.close()
    print("MAPE boxplot saved to 'mape_comparison.pdf'.")
