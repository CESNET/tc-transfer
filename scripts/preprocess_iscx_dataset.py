import os
import pandas as pd

def list_parquet_files(path: str):
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith(".parquet"):
            yield os.path.join(path, entry.name)

dataset_path = "/Downloads/Dataset_ISCX_VPN-nonVPN"
files = sorted(list_parquet_files(dataset_path))
dfs = [pd.read_parquet(file) for file in files]
dataset = pd.concat(dfs).reset_index()

(dataset["iat"].map(len) == dataset.packet_dir.map(len)).all()
(dataset["iat"].map(len) == dataset["L4_payload_bytes"].map(len)).all()

dataset = dataset[dataset.packet_dir.map(len) > 0] # filter flows with empty PPI
dataset_columns = ["packet_dir", "L4_payload_bytes", "iat", "BF_label_app", "BF_label_class", "BF_label_vpn"]
dataset = dataset[dataset_columns]
dataset.to_parquet("ISCXVPN2016.parquet")