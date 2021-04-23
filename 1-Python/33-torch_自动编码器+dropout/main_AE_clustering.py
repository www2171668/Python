import numpy as np 
import pandas as pd
import torch

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import metrics

from train_simple_ae import TrainAE

"""1、读取文件"""
train_file_path = r'F:\xiaoxiang_proj\pytorch_learning\dataset\digit-recognizer\train.csv'
df = pd.read_csv(train_file_path)
print(df.shape)

"""2、数据划分"""
y = df['label'].values
X = df.drop(['label'],1).values

"""3、准备tensor的数据"""
torch_X = torch.from_numpy(X).type(torch.float32)
torch_y = torch.from_numpy(y).type(torch.LongTensor) # data type is long

"""
    -》metrics.adjusted_mutual_info_score(*,*)：计算相关系数
"""
def evaluate(reduced_data):
    # 对聚类算法得到的数据进行评估
    kmeans = KMeans(n_clusters=10)   # 手写体识别，共10个类别
    kmeans.fit(reduced_data)

    nmi = metrics.adjusted_mutual_info_score(y, kmeans.labels_)   # 计算真实y和kmeans算法预测得到的labels_的相关系数
    return nmi


if __name__ == '__main__':
    # 方法一，用聚类方法对原始数据进行评价
    nmi_org = evaluate(X)   # 直接对28*28的原始数据进行类别评价
    print(nmi_org)

    # 方法二，用聚类方法对自编码器得到的数据进行评价
    myAE = TrainAE()
    embedding = myAE.start_train(torch_X)   # 先对数据进行自编码器的训练，然后对隐含层的输出进行类别评价
    nmi_AE = evaluate(embedding)
    print(nmi_AE)
