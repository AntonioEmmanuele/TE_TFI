#!/bin/bash
win_clust=50
general_dir="experiments/ucr_preprocess"
if [ ! -d $general_dir ]; then
  mkdir -p $general_dir
  echo "Directory created."
fi
out_dir="experiments/ucr_preprocess/te_tfi/"
for file in ./datasets/processed/*; do
    python3 ./hyp_cluster.py --cluster_min 2 --cluster_max 10 --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --win_clust $win_clust --series_path $file --out_path $out_dir
    git add *
    git commit -m "adds CLUST Series: $file N_clust : $n_clust Perc : $perc_clust"
    git push
done

