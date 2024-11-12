from lib.ucr_parser import ucr_process_all_ts
import os
if __name__ == "__main__":
    in_path = "datasets/UCR_Anomaly_FullData"
    out_path = "datasets/processed"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    ucr_process_all_ts(in_path = in_path, out_path = out_path)
    