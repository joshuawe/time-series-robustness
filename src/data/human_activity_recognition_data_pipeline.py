
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class HARDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)
    
    def get_num_features(self) -> int:
        return self.x.shape[2]
    
def preprocessing_features(features_list: list[str]) -> np.ndarray:
    dataset = []
    for features in features_list:

        try:
            dataset.append(pd.read_csv(features, sep="\s+", header=None))
        except:
            raise FileNotFoundError(f"{features} not found")
        
    dataset = np.dstack(dataset)
    return dataset

def load_datasets_from_raw(root, dataset_type) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_files = [f"{root}/{dataset_type}/Inertial Signals/body_acc_x_{dataset_type}.txt", 
                    f"{root}/{dataset_type}/Inertial Signals/body_acc_y_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/body_acc_z_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/body_gyro_x_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/body_gyro_y_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/body_gyro_z_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/total_acc_x_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/total_acc_y_{dataset_type}.txt",
                    f"{root}/{dataset_type}/Inertial Signals/total_acc_z_{dataset_type}.txt"]
    x = preprocessing_features(dataset_files)
    y = pd.read_csv(f"{root}/{dataset_type}/y_{dataset_type}.txt", sep="\s+", header=None).to_numpy()

    assert x.shape[0] == y.shape[0], f"{dataset_type}: Number of samples and labels should be same"
    assert x.shape[1] == 128, f"{dataset_type}: Timestamps should be 128"
    assert x.shape[2] == 9, f"{dataset_type}: Number of features should be 9"

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.type(torch.float32), y.type(torch.long)-1

    # Normalize data
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    normalized_data = (x - mean) / std

    return normalized_data, y

def save_postprocessed_dataset(root: str, save_path: str):
    dataset_types = ["train", "test"]
    for dataset_type in dataset_types:
        x, y = load_datasets_from_raw(root, dataset_type)
        with h5py.File(f"{save_path}/{dataset_type}.h5", "w") as f:
            f.create_dataset(f"data", data=x)
            f.create_dataset(f"labels", data=y)

def load_postprocessed_dataset(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {}
    for dataset_type in ["train", "test"]:
        try:
            dataset = h5py.File(f"{config['dataset_path']}/{dataset_type}.h5", "r")
        except:
            raise FileNotFoundError(f"{config['dataset_path']}/{dataset_type}.h5 not found")
        
        x = torch.from_numpy(dataset["data"][:]).to(device)
        y = torch.from_numpy(dataset["labels"][:]).to(device)

        datasets[f"{dataset_type}"] = HARDataset(x, y)
    return datasets['train'], datasets['test']