import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



''' LSTM的相关配置文件 '''
class config(object):
    def __init__(self):
        # 路径相关设置
        self.data_path = ""                                     # 数据的顶层路径
        self.train_data = self.data_path + ""                   # 训练集
        self.eval_data = self.data_path + ""                    # 验证集
        self.test_data = self.data_path + ""                    # 测试集
        self.model_name = "BiLSTM"                              # 模型名称

        # LSTM网络结构设置
        self.hidden_size = 128                                  # 隐藏层
        self.layer_nums = 2                                     # LSTM层数
        self.dropout = 0.3                                      # 随机丢弃
        self.embedding_size = 128                               # 最后将语音嵌入的维度
        self.bidirectional = True                               # 是否双向
        self.frame_num = 128                                    # 读取的梅尔波普矩阵的帧的个数
        self.frame_len = 180                                    # 读取的梅尔波普矩阵的每一帧的长度
        # TODO：补充每个人音频数
        self.sample_len = 15                                    # 每个人的音频数
        # TODO: 修改范围 200人的话 可以修改为5000
        self.negative_num = 50                                   # 每个人对应的负样本，约为 ((sample_len * batch_size) ** 2) / 10


        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.maxiter_without_improvement = 1000                                      # 若超过1000轮效果仍然没有提升，则提前结束训练
        self.epoch = 1000                                                            # 训练轮数
        self.learning_rate =1e-5                                                     # 学习率

        # dataloader部分
        self.batch_size = 2                                   # 每次取出的数据的量

        # 模型保存路径
        self.save_model_path = "./train/models/" + self.model_name + ".ckpt"
        self.log_path = "./train/logs/" + self.model_name + '/'


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.LSTM = nn.LSTM(config.frame_len, config.hidden_size, num_layers=config.layer_nums,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        if config.bidirectional == True:
            self.Fc = nn.Linear(2*config.hidden_size, config.embedding_size)
        else:
            self.Fc = nn.Linear(config.hidden_size, config.embedding_size)


    def forward(self, x):
        output_wav1 = self.LSTM(x)                   # output.shape [batch_size, seq_len, 2*hidden_size] = [batch_size, 32, 64]
        output_wav1 = self.Fc(output_wav1[0][:,-1,:])   # 取最后时刻的隐藏状态 hidden state
        return output_wav1

