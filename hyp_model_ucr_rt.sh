#!/bin/bash
win_clust=50
out_dir="hyp_rt_ucr"
for file in ./datasets/processed/*; do
    python3 ./hyp_models.py --model "RT" --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --win_size $win_clust --series_path $file  --lag_percentage 1.0 --out_path $out_dir
done

