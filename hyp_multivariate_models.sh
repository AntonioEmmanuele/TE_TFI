
#!/bin/bash
out_path="experiments/traffic/xgb"
if [ ! -d $out_path ]; then
    mkdir -p $out_path
fi
for file in ./datasets/traffic/*; do
	python3 ./hyp_models.py --model "XGB" --win_size 24 --series_path ${file}  --lag_percentage 1.0 --out_path $out_path --is_multivariate 1 --target_column "Vehicles" 
done