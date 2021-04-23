import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mlp_model import MLP, evaluate
from sklearn.model_selection import train_test_split

"""手写字体识别"""


"""1、读取文件"""
train_file_path = r'..\..\dataset\digit-recognizer\train.csv'
df = pd.read_csv(train_file_path)
# print(df.shape)

"""2、数据划分"""
X = df[df.columns[1:]].values
y = df[df.columns[0]].values   # 'Label'
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""3、准备tensor的数据 ★"""
# create feature and targets tensor for train set.
torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)   # 将numpy转tensor进行pytorch模型的训练
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

"""4、准备 MBGD 数据 ★★"""
BATCH_SIZE = 32
# ①、torch.utils.data.TensorDataset()：数据封装  Pytorch train and test sets
train_data = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test_data = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
# ②、torch.utils.data.DataLoader(train_data,batch_size)：  data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

print(len(train_data))  # 29400  总样本数
print(len(train_loader))  # 919  单批样本数
print(len(train_loader.dataset))  # 29400  总样本数

"""5、模型训练"""
def train_model(model, train_loader):
    # ①、优化与损失   ★
    # 不同的优化方法的效率，参数可能差别很大
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #optimizer = torch.optim.RMSprop(model.parameters())
    #optimizer = torch.optim.Adadelta(model.parameters())
    #optimizer = torch.optim.Adagrad(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())

    loss_func = nn.CrossEntropyLoss()   # classification

    # ②、模型训练
    EPOCHS = 5
    for epoch in range(EPOCHS):
        correct = 0
        for batch_index, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()   # 清零
            # forward + backward + optimize
            output = model(X_batch.float())   # -》*.float()  转float类型的torch数据； 有时候也会用float(*)
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1]   # 获取当前批次每一份样本的概率最大项的index（预测标签） ★
            correct += (predicted == y_batch).sum()   # 记录每一次迭代的准确预测数量
            #print(correct)
            if batch_index % 100 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'   # {:.6f}：保留小数点后6位
                      .format( epoch, batch_index*len(X_batch), len(train_loader.dataset), 100.*batch_index / len(train_loader),
                               loss.data.item(), 100.*float(correct) / float(BATCH_SIZE*(batch_index+1))))   # Accuracy = 到当前批次准确预测的数量 / 到当前批次所有样本数
                # -》loss.data.item() 损失值 ★
            # Epoch : 0 [6400/29400 (22%)]	Loss: 0.297520	 Accuracy:81.095%

if __name__ == '__main__':
    myModel = MLP()   # 加载其他文件中的网络模型
    print(myModel)
    train_model(myModel, train_loader)   # 模型训练，传入 模型myModel 和 训练数据train_loader
    evaluate(myModel, test_loader, BATCH_SIZE)   # 测试集评价
