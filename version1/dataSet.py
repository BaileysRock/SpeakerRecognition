import json
import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset


class myTrainDataSet(Dataset):
    """
    对Dataset的继承
    """
    def __init__(self, dataset):
        # 定义数据集
        self.dataset = dataset
        # 定义length
        self.length = len(self.dataset)
        assert len(set([len(sample) for sample in self.dataset])) == 1
        # 定义每人的数据集的大小
        self.sample_len = list(set([len(sample) for sample in self.dataset]))[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        random_index = random.randint(0, self.length-1)
        while random_index == item:
            random_index = random.randint(0, self.length-1)
        return {
            'sample': self.dataset[item],
            'negative': self.dataset[random_index]
        }



class myEvalDataSet(Dataset):
    """
    对Dataset的继承
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return {
            'sample_wav1': self.dataset[item][0],
            'sample_wav2': self.dataset[item][1]
        }









