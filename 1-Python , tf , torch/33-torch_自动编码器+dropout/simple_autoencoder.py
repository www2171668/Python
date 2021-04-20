import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    '''Inputs: in_dim, h_dim_1
    '''
    def __init__(self, in_dim=256, h_dim_1=128, dropout=0.5):  # 使用 forward网络构造方法 + 快速构造方法
        super(AutoEncoder, self).__init__()

        # ①、encoder
        self.encoder = nn.Sequential(   # 注意使用的是nn.Sequential()方法，该层全部操作均在其中完成
            nn.Linear(in_dim, h_dim_1),
            nn.LeakyReLU(),
        )
        # ②、decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_dim_1, in_dim)
        )
        self.dropout = dropout   # 先用构造函数接收了dropout的比例


    def forward(self, X):
        embedding = self.encoder(X)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)   # 必须使用training=self.training，以保存train的状态
        out = self.decoder(embedding)
        return embedding, out
