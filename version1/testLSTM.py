import numpy as np
import os

import torch
import torch.nn.functional as F
from LSTM import config
from LSTM import trainModel
from LSTM import testModel
from dataLoader import myDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train

myConfig = config()

# 初始化模型
model = testModel(myConfig).to(myConfig.device)
model.load_state_dict(torch.load(myConfig.save_model_path))


Dict = {

}

# 输入为音频矩阵
def addPerson(data,label):
    if label in Dict.keys():
        print('The person is exist!')
        return
    else:
        Dict[label] = model(data).numpy()
        print("Success!")

# 声纹识别部分
def findPerson(data):
    processData = model(data)
    minCost = float('inf')
    label = None
    for key in Dict.keys():
        iterData = torch.FloatTensor(Dict[key])
        if F.l1_loss(processData,iterData).item() < minCost:
            minCost = F.l1_loss(processData,iterData).item()
            label = key
    return label







