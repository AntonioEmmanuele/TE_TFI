#!/bin/bash
clusters=(2 3 4 5 6 7 8 9 10)
percentages=( 0.5 1.0 2.0 3.0)
win_clust=50
out_dir="hyp_test_multivariate"
for file in ./datasets/etth/*; do
    for n_clust in "${clusters[@]}"; do
        for perc_clust  in "${percentages[@]}"; do
            python3 ./hyp_cluster.py --path_stagionality "./tst/seasonality_results.csv" --num_cluster $n_clust --win_clust $win_clust --series_path $file  --lag_percentage $perc_clust --out_path $out_dir --is_multivariate 1
        done
    done
done

for file in ./datasets/traffic/*; do
    for n_clust in "${clusters[@]}"; do
        for perc_clust  in "${percentages[@]}"; do
            python3 ./hyp_cluster.py --path_stagionality "./tst/seasonality_results.csv" --num_cluster $n_clust --win_clust $win_clust --series_path $file  --lag_percentage $perc_clust --out_path $out_dir --is_multivariate 1
        done
    done
done
