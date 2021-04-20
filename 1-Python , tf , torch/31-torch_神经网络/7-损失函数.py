import numpy as np
import torch
import torch.nn.functional as F

A = np.array([[1,2], [3,4]])
B = np.array([[1,4], [2,4]])

cost_np = np.sum((A-B)**2)
print('Numpy cost is %f' % cost_np)

A = torch.FloatTensor(A)
B = torch.FloatTensor(B)


'''------------- 适用于回归问题 -------------'''
# mse loss: measures the mean squared error (squared L2 norm) between each element in the input x and target y.
cost_torch = F.mse_loss(A, B, reduction='sum') # reduction: ‘none’ | ‘mean’ | ‘sum’
print('Torch cost is ', cost_torch)
# l1_loss: measures the mean absolute error (MAE) between each element in the input x and target y.
cost_torch = F.l1_loss(A, B, reduction='sum')
print('Torch cost is ', cost_torch)


'''------------- 适用于分类问题 -------------'''
# input is of size N x C = 3 x 5
input = torch.ones(3, 5, requires_grad=True)
print('input value: ', input)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])

print('softmax', F.softmax(input))

temp = F.log_softmax(input)
print(temp)
output = F.nll_loss(temp, target)
print("nll loss", output)

output = F.cross_entropy(input, target)
print("cross entropy loss", output)
