import numpy as np

from LSTM import config
from LSTM import trainModel
from dataLoader import myDataSet
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from Mel import get_mel_feature

if __name__ == "__main__":
    # 生成相关配置
    myConfig = config()

    SAMPLE_RATE = 44100

    # TODO:把所有训练集和验证集处理成如下格式 data = [(wav11,wav12,label1),(wav21,wav22,label2),...], wav11为音频1的梅尔波普，wav12为音频2的梅尔波普
    # 补充dataTrain dataEval
    dataTrain = [
        (get_mel_feature("train-clean-100/19/198/19-198-0000.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0001.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0002.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0003.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0004.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0005.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0006.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0007.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0008.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0009.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0010.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0011.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0012.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0013.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0014.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0015.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0016.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0017.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0018.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0019.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0020.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0021.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0022.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0023.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0024.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0025.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0026.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0027.flac", SAMPLE_RATE), 1),
        (get_mel_feature("train-clean-100/19/198/19-198-0028.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/198/19-198-0029.flac", SAMPLE_RATE), 1),

        (get_mel_feature("train-clean-100/26/495/26-495-0000.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0001.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0002.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0003.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0004.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0005.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0006.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0007.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0008.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0009.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0010.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0011.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0012.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0013.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0014.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0015.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0016.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0017.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0018.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0019.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0020.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0021.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0022.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0023.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0024.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0025.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0026.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0027.flac", SAMPLE_RATE), 2),
        (get_mel_feature("train-clean-100/26/495/26-495-0028.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/495/26-495-0029.flac", SAMPLE_RATE), 2),
    ]
    dataEval = [
        (get_mel_feature("train-clean-100/19/227/19-227-0000.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/19/227/19-227-0001.flac", SAMPLE_RATE), 1),

        (get_mel_feature("train-clean-100/26/496/26-496-0000.flac", SAMPLE_RATE), get_mel_feature("train-clean-100/26/496/26-496-0001.flac", SAMPLE_RATE), 2),
    ]
    trainDataSet = myDataSet(dataTrain)
    evalDataSet = myDataSet(dataEval)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    evalDataLoader = DataLoader(dataset=evalDataSet,batch_size=myConfig.batch_size)

    # 初始化模型
    model = trainModel(myConfig).to(myConfig.device)

    train(myConfig,model,trainDataLoader,evalDataLoader)




