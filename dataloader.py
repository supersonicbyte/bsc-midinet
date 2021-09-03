import numpy as np
import torch
from torch.utils.data import Dataset

class BarDataset(Dataset):
    def __init__(self, data, data_prev, device):
        self.device = device
        self.data, self.data_prev, self.data_len = self.extract_data(data, data_prev)
    
    def __len__(self):
        return self.data_len

    @property
    def shape(self):
        return self.data.shape
    
    def __getitem__(self, idx):
        X = self.data[idx]
        X_prev = self.data_prev[idx]
        return X, X_prev
    
    def extract_data(self, data, data_prev):
        data_len = data.shape[0]
        data = torch.Tensor(data).to(self.device)
        data_prev = torch.Tensor(data_prev).to(self.device)
        return data, data_prev, data_len