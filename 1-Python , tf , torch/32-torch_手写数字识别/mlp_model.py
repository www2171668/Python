import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    网络文件
'''

class MLP(nn.Module):   # 构建神经网络
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 250) # image width and height are both 28, 28*28=784
        self.linear2 = nn.Linear(250, 100) # map 250-d to 100-d
        self.linear3 = nn.Linear(100, 10) # map 100-d to 10-d, since there are 10 classes (0,1,..,9)
    
    def forward(self, x):
        x = F.relu(self.linear1(x)) # the first hidden layer
        x = F.relu(self.linear2(x)) # the second hidden layer
        x = self.linear3(x) # the final output layer
        return x

def evaluate(model, test_loader, BATCH_SIZE):   # 测试集评估方法
    correct = 0
    for test_imgs, test_labels in test_loader:   #  ★
        output = model(test_imgs.float())

        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( 100*float(correct) / (len(test_loader)*BATCH_SIZE)))   # len(train_loader.dataset) = len(test_loader)*BATCH_SIZE；但是在方法中无法使用.dataset
