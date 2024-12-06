#!/bin/bash
win_clust=50
out_dir="hyp_test_ucr_new"
for file in ./datasets/processed/*; do
    python3 ./hyp_cluster.py --cluster_min 2 --cluster_max 10 --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --win_clust $win_clust --series_path $file --out_path $out_dir
    # git add *
    # git commit -m "adds Series: $file N_clust : $n_clust Perc : $perc_clust"
    # git push
done

