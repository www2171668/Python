import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from simple_autoencoder import AutoEncoder

# 训练一个最朴素的AE，
class TrainAE():
    def __init__(self):
        self.epochs = 10
        self.dropout = 0.5 # test this
        
    def start_train(self, X):        
        # Model and optimizer
        _, d = X.shape
        self.model = AutoEncoder(in_dim=d, dropout=self.dropout)   # 让图像属性(维度)作为输入层神经元数量in_dim，而让隐藏层神经元数量h_dim为默认值
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            self.model.train()   # ①、在模型训练时，将model设置到train模式，使dropout起作用  ★

            optimizer.zero_grad()

            # forward + backward + optimize
            embedding, output = self.model(X)
            loss_train = F.mse_loss(X, output, size_average=False)   # 回归
            print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()), end="\n")
            loss_train.backward()
            optimizer.step()

        return self.get_latent_embedding(X)

    def get_latent_embedding(self, X):
        self.model.eval()   # ②、在模型评估时，将model设置到eval模式，此时框架会自动把BN和DropOut固定住，使用训练好的值，相当于使用完整的网络  ★
        embedding, output = self.model(X)
        return embedding.data.numpy()   # 返回隐含层的输出
