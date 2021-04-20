import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, test_loader, BATCH_SIZE):
    correct = 0 
    for test_imgs, test_labels in test_loader:
        output = model(test_imgs.float())
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()

    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))

# Defining the network (LeNet-5)  
class LeNet5(torch.nn.Module):
    def __init__(self):   
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input ，然而我们现有的数据是28*28. Hence padding of 2 is done below)
        # output size = ((28+4)-5 + 1)/1 = 28  1@28*28 ----> 6@28*28
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) #输出featuremap大小：28*28，经过padding 28+4=32，实际是对32*32的图像进行处理
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)# Max-pooling，input 6@28*28 ---> 6@14*14

        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0) # input 6@14*14 ----> 16@10*10
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)# Max-pooling，input 16@10*10 ---> 16@5*5

        # Fully connected layers
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x)) # 第一步卷积
        x = self.max_pool_1(x)                      # 第二步池化

        x = torch.nn.functional.relu(self.conv2(x)) # 第三步卷积
        x = self.max_pool_2(x)                      # 第四步池化

        # 展平 详见 https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)                      # 第五步，把二维的矩阵转化成一维的vector
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
