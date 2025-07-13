# sdf_dataset.py --------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    """
    Loads points + signed distances stored in a .npz file created by make_dataset.py
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.points = torch.from_numpy(data["points"])  # (N,3) float32
        self.sdf    = torch.from_numpy(data["sdf"])     # (N,)  float32

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.sdf[idx]
