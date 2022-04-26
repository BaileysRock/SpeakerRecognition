import numpy as np

from LSTM import config
from LSTM import trainModel
from dataLoader import myDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train

if __name__ == "__main__":
    # 生成相关配置
    myConfig = config()

    # TODO:把所有训练集和验证集处理成如下格式 data = [(wav11,wav12,label1),(wav21,wav22,label2),...], wav11为音频1的梅尔波普，wav12为音频2的梅尔波普
    # 补充dataTrain dataEval
    dataTrain = [(np.random.random((140,120)),np.random.random((140,120)),1),(np.random.random((140,120)),np.random.random((140,120)),1)]
    dataEval = [(np.random.random((140,120)),np.random.random((140,120)),1),(np.random.random((140,120)),np.random.random((140,120)),1)]
    trainDataSet = myDataSet(dataTrain)
    evalDataSet = myDataSet(dataEval)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    evalDataLoader = DataLoader(dataset=evalDataSet,batch_size=myConfig.batch_size)

    # 初始化模型
    model = trainModel(myConfig).to(myConfig.device)

    train(myConfig,model,trainDataLoader,evalDataLoader)




