#!/bin/bash
win_clust=50
out_dir="hyp_randomforest_ucr"
files=()
offset=0 # Used for experiments error
end_offset=250 # Used for splitting experiments
# Add each file in the directory to the array
for file in ./datasets/processed/*; do 
  # Check if it's a regular file
  if [ -f "$file" ]; then
    files+=("$file")
  fi
done

# Iterate over the range from start_index to end_index
for ((i=offset; i<end_offset; i++)); do
  	echo "Processing: ${files[i]}"
	python3 ./hyp_models.py --model "XGB" --path_stagionality "./tst/ucr_seasonality_results_0.6.csv" --win_size $win_clust --series_path ${files[i]}  --lag_percentage 1.0 --out_path $out_dir
	git add *
	git commit -m "adds XGBoost ${files[i]}"
	git push
done