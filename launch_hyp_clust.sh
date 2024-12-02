#!/bin/bash
clusters=(2 3 4 5 6 7 8 9 10)
percentages=( 0.5 1.0 2.0 3.0)
win_clust=50
out_dir="hyp_test_ucr"
for file in ./datasets/processed/*; do
    for n_clust in "${clusters[@]}"; do
        for perc_clust  in "${percentages[@]}"; do
        #    echo "$file"
            python3 ./hyp_cluster.py --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --num_cluster $n_clust --win_clust $win_clust --series_path $file  --lag_percentage $perc_clust --out_path $out_dir
    done
  done
done

