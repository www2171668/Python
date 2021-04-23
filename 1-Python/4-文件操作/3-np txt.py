""""""
# %% 1:numpy.savetxt('fname',X):第一个参数为文件名，第二个参数为需要存的数组
# ! 2.numpy.loadtxt('fname')：将数据读出为array类型
import numpy as np

l1 = np.arange(5)
l2, l3 = l1 * 2, l1 * 3
np.savetxt('data/001.txt', (l1, l2, l3))
a = np.loadtxt('data/001.txt')
print(a)
