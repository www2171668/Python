import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from cnn_model import LeNet5, evaluate
from sklearn.model_selection import train_test_split

"""1、读取文件"""
train_file_path = r'C:\vscode_proj\dataset\digit-recognizer\train.csv'
df = pd.read_csv(train_file_path)
# print(df.shape)

"""2、数据划分"""
X = df[df.columns[1:]].values
y = df[df.columns[0]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""3、准备tensor的数据"""
torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

# ----- for cnn -------
torch_X_train = torch_X_train.view(-1, 1,28,28).float()   # 修改图像数据格式 [batch_size, channels, height, weight] ★
torch_X_test = torch_X_test.view(-1,1,28,28).float()

"""4、准备 MBGD 数据"""
BATCH_SIZE = 32

train_data = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test_data = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

"""5、模型训练"""
def train_model(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()

    EPOCHS = 5
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_batch.float())
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == y_batch).sum()
            if batch_idx % 100 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'
                      .format( epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader),
                               loss.data.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))

if __name__ == '__main__':
    myModel = LeNet5()
    print(myModel)
    train_model(myModel, train_loader)
    evaluate(myModel, test_loader,BATCH_SIZE)
