import pathlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from .data_normalizer import Normalizer

class DatasetPlanetarySystems(Dataset):
    def __init__(self, path: pathlib.Path):
        df = pd.read_csv(path).sort_values(["system_number", "a"]).copy()
        
        self.df = df
        self.normalizer = Normalizer()
        self.normalizer.fit(df)
        self.normalized = self.normalizer.transform(df)
        
        
        self.groups = [
            torch.tensor(
                g[["a","total_mass","r"]].to_numpy(), dtype=torch.float32)
                for _, g in self.normalized.groupby("system_number")
                ]

    def __len__(self): return len(self.groups)
    def __getitem__(self, idx): return self.groups[idx]