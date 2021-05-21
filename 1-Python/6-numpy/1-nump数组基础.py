""""""
import numpy as np

# %% np.array() / np.asarray(a)   将列表、元组、列表元祖等转数组
a = [1, 2, 3, 4, 5]  # ! 或 a = (1, 2, 3, 4, 5)
arr = np.array(a)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# %% np.empty(()) / np.ones(()) / np.zeros(())  创建空数组，注意内层()          dtype 控制数组元素的类型，il为整数
arrs = np.empty((2, 3), dtype='i1')
arrs = np.ones((2, 3))
arrs = np.zeros((2, 3))

# %% np.identity(n) 标准单位矩阵
c = np.identity(3)
print(c)

# %% *.dtype 返回数组中元素的类型。区别于type()       *.astype(*) 数据类型转换
a = np.random.randint(1, 6, size=(2, 3))
print(type(a))
print(a.dtype)

b = a.astype(np.float64)  # \ int转float
print(b.dtype)

# %% np.arange(*)  创建规则数组  range函数的数组版，返回的是一个ndarray，而不是list
a = np.arange(9)  # \[0 1 2 3 4 5 6 7 8]
b = np.arange(1, 9, 2)  # \[1 3 5 7]

# %% 空维度
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [2, 1, 3]])
print(a.shape)
print(b.shape)
print(len(a), len(b))

a.shape = (-1, 1)  # \ 将(3,)向量转为(3,1)
print(a.shape)

# %% np.linspace(起始数字，结束数字，数列数量, retstep) 等差数列     retstep=True时，显示等差距离
arr = np.linspace(0, 10, 5)
print(arr)  # \[ 0.   2.5  5.   7.5 10. ]

# %% np.logspace(起始指数，结束指数，数列数量) 等比数列     起始位和终止位代表的是10的幂（默认基数为10）    base改变幂指数的基数。不添加时默认为10
arr = np.logspace(0, 4, 3)
print(arr)  # \ 1 100 10000
arr = np.logspace(0, 4, 2, base=2)
print(arr)

# %% 创建随机数组
# * np.random.random((X, Y)) / np.random.rand(X, Y)  创建[0, 1)之间的随机数
# * np.random.randint(X, Y)   返回X行Y列的随机整数数组  —— size 控制矩阵形状
# * np.random.randn(X, Y)   返回[0, 1)之间的标准正态分布随机样本

arr = np.random.random((2, 3))
arr = np.random.randn(2, 3)

arr = np.random.randint(0, 2)  # * 1、np.random.randint 前包后不包
# arr= np.random.randint(1,9,size = (3,3))
print(arr)

import random  # * 2、前后都包

arr = random.randint(0, 2)
print(arr)

from random import randint  # * 3、前后都包

arr = randint(0, 2)
print(arr)

# %% np.random.seed(*) 产生种子号
np.random.seed(100)
arr = np.random.randint(1, 9, size=(3, 3))
print(arr)
