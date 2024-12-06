#!/bin/bash
win_clust=50
out_dir="hyp_randomforest_ucr"
for file in ./datasets/processed/*; do
    python3 ./hyp_models.py --model "RF" --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --win_size $win_clust --series_path $file  --lag_percentage 1.0 --out_path $out_dir
    git add *
    git commit -m "adds Random Forest ${file}"
    git push
done

