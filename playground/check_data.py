import pandas as pd
cur_task="7ddd70cc8e6740fc8ca4b7a59577fe37"
df = pd.read_parquet(f"/mnt/fastdisk/output/{cur_task}/data/chunk-000/file-000.parquet")

print(df.head())


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

# dataset path
dataset = LeRobotDataset(f"/mnt/fastdisk/output/{cur_task}")

# dataloader
loader = DataLoader(dataset, batch_size=1, shuffle=False)

batch = next(iter(loader))

print("Batch keys:", batch.keys())

for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, type(v))