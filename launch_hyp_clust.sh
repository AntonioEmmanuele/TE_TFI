#!/bin/bash
python3 ./hyp_cluster.py --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --num_cluster 5 --win_clust 50 --series_path "./datasets/processed/011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100.txt" --lag_percentage 0.5
