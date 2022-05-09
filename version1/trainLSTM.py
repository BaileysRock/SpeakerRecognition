from LSTM import config
from LSTM import Model
from dataSet import trainDataLoader
from dataSet import evalDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from Mel import load_mel_feature
import numpy as np


def load_data():
    """
    从文件中加载训练集和测试集

    Returns:
        训练集为如下格式
        data = [
            (wav11, wav12, label1),
            (wav21, wav22, label2),
            ...
        ]

        测试集为如下格式
        data = [
            (wav11, wav12),
            (wav21, wav22),
            ...
        ]

        其中 wav11 为音频 1 的 MFCC 特征矩阵，wav12 为音频 2 的 MFCC 特征矩阵
    """

    dataTrain = []
    dataEval = []

    feature_label = load_mel_feature(human_count=200)

    d = {}
    for fl in feature_label:
        if fl[1] not in d:
            d[fl[1]] = [fl[0]]
        else:
            d[fl[1]].append(fl[0])

    for l in d.keys():
        for i in range(0, len(d[l]) - 4, 2):
            dataTrain.append(
                (d[l][i], d[l][i + 1], l)
            )
        dataEval.append(
            (d[l][len(d[l]) - 2], d[l][len(d[l]) - 1])
        )

    return dataTrain, dataEval


def random_data():
    """
    从文件中加载训练集和测试集

    Returns:
        训练集为如下格式
        data = [
            (wav11, wav12, label1),
            (wav21, wav22, label2),
            ...
        ]

        测试集为如下格式
        data = [
            (wav11, wav12),
            (wav21, wav22),
            ...
        ]

        其中 wav11 为音频 1 的 MFCC 特征矩阵，wav12 为音频 2 的 MFCC 特征矩阵
    """

    dataTrain = [(np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))), (
                     np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                     np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                     np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 (np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180)), np.random.random((128, 180)),
                  np.random.random((128, 180)), np.random.random((128, 180))),
                 ]

    dataEval = [(np.random.random((128, 180)), np.random.random((128, 180))),
                (np.random.random((128, 180)), np.random.random((128, 180))),
                (np.random.random((128, 180)), np.random.random((128, 180))),
                (np.random.random((128, 180)), np.random.random((128, 180))),
                (np.random.random((128, 180)), np.random.random((128, 180)))]

    return dataTrain, dataEval


if __name__ == "__main__":
    # 生成相关配置
    myConfig = config()

    # dataTrain, dataEval = load_data()
    dataTrain, dataEval = random_data()

    trainDataLoader = trainDataLoader(dataTrain, myConfig)
    evalDataSet = evalDataSet(dataEval)

    evalDataLoader = DataLoader(dataset=evalDataSet, batch_size=myConfig.batch_size, shuffle=False)

    # 初始化模型
    model = Model(myConfig).to(myConfig.device)

    train(myConfig, model, trainDataLoader, evalDataLoader)

    # trainDataSet = myTrainDataSet(dataTrain)
    # evalDataSet = myEvalDataSet(dataEval)
    # trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    # evalDataLoader = DataLoader(dataset=evalDataSet, batch_size=myConfig.batch_size, shuffle=False)
    #
    # # 初始化模型
    # model = trainModel(myConfig).to(myConfig.device)
    #
    # train(myConfig, model, trainDataLoader, evalDataLoader)
