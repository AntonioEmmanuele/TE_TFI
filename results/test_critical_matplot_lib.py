

import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
import pandas as pd
# Updated Data
models = ['RandomForest', 'XGBoost', 'Ours', 'RegressionTree']
mean_ranks = [1.644068, 1.813559, 2.881356, 3.661017]
nemenyi_pvals = np.array([
    [1.000000e+00, 3.221657e-10, 0.000000, 0.000000],
    [3.221657e-10, 1.000000e+00, 0.000000, 0.000000],
    [0.000000e+00, 0.000000, 1.000000e+00, 0.482903],
    [0.000000e+00, 0.000000, 0.482903, 1.000000]
])
#cd = 0.5  # Set the critical difference
#cd = compute_CD(mean_ranks, 236, 0.05, "nemenyi")
cd = 0.432
# Re-plotting Critical Difference Diagram
# Re-plotting without names on the upper part
plt.figure(figsize=(10, 4))
plt.hlines(1, min(mean_ranks) - 0.5, max(mean_ranks) + 0.5, color='black', linestyles='--', linewidth=0.7)

# Draw the models and their mean ranks
for i, (model, rank) in enumerate(zip(models, mean_ranks)):
    plt.plot(rank, 1, 'o', markersize=10, label=model)

# Draw connections for models within the critical difference
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        if abs(mean_ranks[i] - mean_ranks[j]) <= cd:
            plt.plot([mean_ranks[i], mean_ranks[j]], [1, 1], color='blue', linewidth=2)

# Customize plot
plt.title("Critical Difference Diagram")
plt.xlabel("Mean Rank")
plt.yticks([])
plt.xticks(np.arange(min(mean_ranks) - 0.5, max(mean_ranks) + 0.5, 0.5))
plt.grid(axis='x', linestyle='--', linewidth=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig("critical_matplotlib.pdf", format ="pdf", dpi = 1600)
plt.close("all")

# graph_ranks(mean_ranks, models, nemenyi_pvals,
#             cd=cd, reverse=True, width=9, textspace=1.5, labels=False)

# # font = {'family': 'sans-serif',
# #     'color':  'black',
# #     'weight': 'normal',
# #     'size': 22,
# #     }

# plt.savefig('cd-diagram.png',bbox_inches='tight')

# # Prepare the data for scikit-posthocs
# data = pd.DataFrame({
#     'Model': models,
#     'Mean Rank': mean_ranks
# })

# # Plot the Critical Difference Diagram using scikit-posthocs
# plt.figure(figsize=(10, 6))
# sp.sign_plot(
#     nemenyi_pvals, 
#     data['Model'], 
#     data['Mean Rank'], 
#     alpha=0.05, 
#     cd=cd
# )
# plt.title("Critical Difference Diagram (scikit-posthocs)")
# plt.savefig("critical_2.pdf", format ="pdf", dpi = 1600)
