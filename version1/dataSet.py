import json
import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

def getLoss(elem):
    return elem[3]


class evalDataSet(Dataset):
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
        return {
            'wav1': torch.FloatTensor(self.dataset[item][0]),
            'wav2': torch.FloatTensor(self.dataset[item][1])
        }


class trainDataLoader(object):
    def __init__(self, dataset, config):
        """
        :param dataset:dataset所有数据集
        :param config: 相关配置
        """
        # batches为数据集
        self.dataset = dataset
        # batch_size为打包的数据大小
        self.batch_size = config.batch_size
        # n_batch为打包的个数
        self.n_batches = len(self.dataset) // self.batch_size
        self.residue = False
        # if len(self.dataset) != self.batch_size*self.n_batches:
        #     self.residue = True
        self.index = 0
        self.device = config.device
        self.negative_num = config.negative_num

    def ToTensor(self, batch_data):
        """
        :param batch_data:batches截取后的段
        :return: tensor
        """
        y1 = []
        y2 = []
        y3 = []
        # 选取合适的三元组的negative数据
        for i in range(len(batch_data)):
            mse_list = [F.mse_loss(torch.FloatTensor(j), torch.FloatTensor(k)).item() for j in batch_data[i] for k in batch_data[i]]
            mse_list.sort()
            mse_standard = mse_list[len(mse_list)//2]
            y_item = []
            for j in range(len(batch_data)):
                if i != j:
                    for index_1 in range(len(batch_data[i])):
                        for index_2 in range(len(batch_data[i])):
                            if index_1 != index_2:
                                for item in batch_data[j]:
                                    loss_nagative = F.mse_loss(torch.FloatTensor(item), torch.FloatTensor(batch_data[i][index_1]))
                                    loss_positive = F.mse_loss(torch.FloatTensor(batch_data[i][index_1]), torch.FloatTensor(batch_data[i][index_2]))
                                    if loss_nagative.item() > mse_standard and loss_positive.item() < mse_standard:
                                        y_item.append([batch_data[i][index_1], batch_data[i][index_2], item, loss_positive.item()])
            # 去除掉过大的数据
            y_item.sort(key=getLoss)
            y_item = y_item[0:self.negative_num]
            y1.extend([item[0] for item in y_item])
            y2.extend([item[1] for item in y_item])
            y3.extend([item[2] for item in y_item])
        y1 = torch.FloatTensor(np.array(y1))
        y2 = torch.FloatTensor(np.array(y2))
        y3 = torch.FloatTensor(np.array(y3))
        return (y1, y2, y3)

    def __next__(self):
        """
        DataLoader迭代器
        """
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches

    def __iter__(self):
        """
        :return:返回本身
        """
        return self

    def __len__(self):
        """
        :return: 返回长度
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches











