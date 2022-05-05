from LSTM import config
from LSTM import trainModel
from dataLoader import myDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from Mel import load_mel_feature
import numpy as np
if __name__ == "__main__":
    # 生成相关配置
    myConfig = config()

    SAMPLE_RATE = 44100

    # 训练集和验证集为如下格式 data = [(wav11,wav12,label1),(wav21,wav22,label2),...], wav11为音频1的梅尔波普，wav12为音频2的梅尔波普
    # dataTrain = []
    # dataEval = []
    #
    # feature_label = load_mel_feature(human_count=100)
    #
    # for i in range(0, len(feature_label) - 2, 2):
    #     if feature_label[i][1] == feature_label[i + 1][1]:
    #         if i % 10 != 0:
    #             dataTrain.append(
    #                 (feature_label[i][0], feature_label[i + 1][0], feature_label[i][1])
    #             )
    #         else:
    #             dataEval.append(
    #                 (feature_label[i][0], feature_label[i + 1][0], feature_label[i][1])
    #             )

    dataTrain = [(np.random.random((128, 180)), np.random.random((128, 180)), 1),
                 (np.random.random((128, 180)), np.random.random((128, 180)), 1)]
    dataEval = [(np.random.random((128, 180)), np.random.random((128, 180)), 1),
                (np.random.random((128, 180)), np.random.random((128, 180)), 1)]

    trainDataSet = myDataSet(dataTrain)
    evalDataSet = myDataSet(dataEval)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    evalDataLoader = DataLoader(dataset=evalDataSet, batch_size=myConfig.batch_size)

    # 初始化模型
    model = trainModel(myConfig).to(myConfig.device)

    train(myConfig, model, trainDataLoader, evalDataLoader)
