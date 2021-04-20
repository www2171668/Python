import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sequence = nn.Sequential(
            nn.Linear(5, 4),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        o = self.fc1(x)
        o = self.relu1(o)
        o = self.fc2(o)
        return o

net = Net()

# Parameter类是在Tensor上封装起来的，也有is_leaf的bool值
# print('parameter start')
# for name, parameter in net.named_parameters():
#     print(name)
#     print(type(parameter))
#     print(parameter.is_leaf)
# print('parameter end')

for tag, value in net.named_parameters():
    # print(tag,value)
    print(value.data)
    print(value.grad)