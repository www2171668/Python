import torch
from torch.nn import functional as F

data_num = 100
x = torch.unsqueeze(torch.linspace(-1,1,data_num), dim=1)   # [-1,1]等差数列，共100个数；1维列向量
y0 = torch.zeros(50) 
y1 = torch.ones(50)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
print(x.data)
print(y.data)

def get_acc(labels, outputs):
    '''get_acc: Get the classification accuracy
        Inputs: labels: the ground truth; 
                outputs: the predicted labels of a DNN 
        Outputs: the accuracy 
    '''
    _, predicted = torch.max(outputs.data, 1)
    data_num = y.shape[0]*1.0
    #item() to get a Python number from a tensor containing a single value:
    correct_num = (predicted == labels).sum().item()
    accuracy = correct_num/data_num
    return accuracy

class SimpleNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):   # 单隐藏层
        super(SimpleNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)   # 输入层->隐藏层
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 隐藏层->输出层
    
    def forward(self, x):
        hidden_result = self.hidden(x)   # 输入数据
        x = F.relu(hidden_result)   # ReLu激活函数
        x = self.predict(x)   # 输出数据
        return x

# 训练一个分类器，二类分类器
mynet = SimpleNet(1,10,2)
print(mynet.parameters)

optimizer = torch.optim.SGD(mynet.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(1000):
    out = mynet(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        acc = get_acc(y, out)
        print('acc is:', acc)
