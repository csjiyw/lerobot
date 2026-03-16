import json

import pandas as pd
cur_task="0044f0e441f35a7eaa39889dd41be09b"
df = pd.read_parquet(f"/mnt/sdc/output/{cur_task}/data/chunk-000/file-000.parquet")

print(df.head())


with open(f"/mnt/sdc/output/{cur_task}/meta/info.json") as f:
    info = json.load(f)

print(info)

with open(f"/mnt/sdc/output/{cur_task}/meta/info.json") as f:
    info = json.load(f)

print(info)



from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

# dataset path
dataset = LeRobotDataset(f"/mnt/sdc/output/{cur_task}")

# dataloader
loader = DataLoader(dataset, batch_size=1, shuffle=False)

batch = next(iter(loader))

print("Batch keys:", batch.keys())

for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, type(v))