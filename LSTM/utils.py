
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        dataset.append(pd.read_csv(features, sep="\s+", header=None))
    dataset = np.dstack(dataset)
    return dataset

def load_datasets(root, dataset_type) -> HARDataset:
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

    dataset = HARDataset(normalized_data, y)
    return dataset

