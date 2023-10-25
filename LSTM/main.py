# %%
from utils import load_datasets
from model import Model
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load datasets
root = "/mnt/c/Users/kalee/Desktop/ML learning/time-series-robustness/datasets/UCI HAR Dataset"
datasets = {}
for dataset_type in ["train", "test"]:
    dataloader = DataLoader(load_datasets(root, dataset_type), batch_size=64)
    datasets[f"{dataset_type}"] = dataloader

# define model
in_features = datasets["train"].get_num_features()
hidden_size = [256, 128]
num_classes = 6

model = Model(datasets, in_features, hidden_size, num_classes).to(device)

# train model
model.train(epochs=10)
#%%

# evaluate model
model.evaluate()