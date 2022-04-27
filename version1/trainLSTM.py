import numpy as np
import os

from LSTM import config
from LSTM import trainModel
from dataLoader import myDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from Mel import get_mel_feature
from datasets import gen_file_paths

if __name__ == "__main__":
    # 生成相关配置
    myConfig = config()

    SAMPLE_RATE = 44100

    # 训练集和验证集为如下格式 data = [(wav11,wav12,label1),(wav21,wav22,label2),...], wav11为音频1的梅尔波普，wav12为音频2的梅尔波普
    dataTrain = []
    dataEval = []

    human_id_list = os.listdir("./dataset/")
    for i in range(len(human_id_list)):
        if int(human_id_list[i]) >= 100:
            print("pass")
            continue
        path = gen_file_paths(human_id_list[i])
        human_id_train = path[:-2]
        human_id_eval = path[-2:]  # len(human_id_eval) == 2
        for j in range(0, len(human_id_train) - 2, 2):
            dataTrain.append(
                (get_mel_feature(human_id_train[j], SAMPLE_RATE), get_mel_feature(human_id_train[j + 1], SAMPLE_RATE), i)
            )
        dataEval.append(
            (get_mel_feature(human_id_eval[0], SAMPLE_RATE), get_mel_feature(human_id_eval[1], SAMPLE_RATE), -1)  # label doesn't matter
        )

    trainDataSet = myDataSet(dataTrain)
    evalDataSet = myDataSet(dataEval)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    evalDataLoader = DataLoader(dataset=evalDataSet,batch_size=myConfig.batch_size)

    # 初始化模型
    model = trainModel(myConfig).to(myConfig.device)

    train(myConfig,model,trainDataLoader,evalDataLoader)




