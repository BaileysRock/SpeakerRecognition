import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class myDataSet(Dataset):
    """
    对Dataset的继承
    """
    def __init__(self, data):
        self.wav1 = [item[0] for item in data]
        self.wav2 = [item[1] for item in data]
        self.label = [item[2] for item in data]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return {
            'wav1': self.wav1[item],
            'wav2': self.wav2[item],
            'label': self.label[item]
        }












