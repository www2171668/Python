""""""
import numpy as np

# %% 数组形状属性
# len(*) 返回数组维度（秩）
# *.shape / np.shape(*) 返回数组的形状；或改变shape   ★★
# *.size / np.size(*) 返回数组元素个数   ★
# *.T / *.transpose() 转置

a = np.random.random((1, 3))
print('数组的维度：', len(a))
print('数组的形状：', a.shape)
print('获取某一维度', a.shape[0], a.shape[1])
print('数组的个数：', a.size)
print(a.T)

# %% *.reshape(*) / *.shape = (*) 修改数组形状        *.resize(*) 用来改变图片大小，会改变图片的数据内容
# 1、.reshape / .shape
a = np.arange(10).reshape(2, 5)
b = np.arange(10).reshape(-1, 5)
print(a)
print(b)

a = np.arange(10)
a.shape = (2, 5)  # 修改数组的形状
print(a)

a.shape = (5, -1)  # * -1：在保证行数为5的情况下，自动匹配新矩阵的列
print(a)
a.shape = (-1, 5)  # * -1：在保证列数为5的情况下，自动匹配新矩阵的行
print(a)
a.shape = (-1,)  # * 转换为一维行向量
print(a)

# 2、 .resize
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

X_new = np.resize(X, (3, 3))
print(X_new)

# %% *.ravel() 将数组降至一维，返回的是数组 ★★★        *.flat 将数组将至一维，返回的是numpy.flatiter,需要转np才能用
# 修改.ravel返回值会修改原始矩阵，修改.flatten返回值不会影响原始矩阵

x = np.array([[1, 2, 3], [2, 3, 4]])
print(x.ravel(), x.ravel().shape)

# %% *.squeeze() 从数组的形状中删除单维度条目，即把shape中为1的维度去掉      np没有unsqueeze()属性
a = np.arange(10).reshape(1, 10)  # 用a = np.array([[[1, 2, 3, 4]]])没用
print(a.shape)
b = np.squeeze(a)
print(b.shape)

# %% np.stack(*) 叠加
# 按行遍历，每完成所有数据的一行遍历，就形成一个数组；最后对这些数组按列遍历进行叠加
# 按列遍历，每完成所有数据的一列遍历，就形成一个数组；最后对这些数组按行遍历进行叠加
a = np.array([1, 2])
b = np.array([5, 6])
X = np.stack((a, b), axis=0)
Y = np.stack((a, b), axis=1)
print(X)
print(Y)

# %%
a = np.arange(1, 10).reshape((3, 3))
b = np.arange(11, 20).reshape((3, 3))
c = np.arange(101, 110).reshape((3, 3))
print(np.stack((a, b, c), axis=0))  # -》axis=0：块状叠加
print(np.stack((a, b, c), axis=1))  # -》axis=1：行状叠加
print(np.stack((a, b, c), axis=2))  # -》axis=2，列状叠加

# %% np.concatenate()  叠加 ★
# axis = 0，按列遍历（跨行），与vstack功能一致.将两个多维数组在竖直方向上堆叠
# axis = 1，按行遍历（跨列），与hstack功能一致。将两个多维数组在水平方向上堆叠
# 第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸。   一般不会用vstack和hstack，因为vstack和hstack必须保证元素个数完全对应，concatenate可以不用这么严格
a = np.array([1, 2])
b = np.array([5, 6])
X = np.concatenate((a, b), axis=0)  # np.vstack((a,b))
print(X)

# %%
a = np.arange(1, 10).reshape((3, 3))
b = np.arange(11, 20).reshape((3, 3))
c = np.arange(101, 110).reshape((3, 3))
X = np.concatenate((a, b, c), axis=0)  # np.vstack((a,b))
Y = np.concatenate((a, b, c), axis=1)  # np.hstack((a,b))
print(X)
print(Y)

# %% np.dstack 元素重组，数组的元素依次对应，维度 + 1 ★★★
X = np.dstack((a, b))
print(X)
# %% 分割 np.split(data,index)   按照index拆分数组，返回拆分后元素组成的新的数组   ★
x = np.arange(9)
np.split(x, 3)  # \ 3个数组为一组，进行分割

y = np.array([[1, 1, 1], [2, 2, 2]])
m, n = np.split(y, (2,), axis=1)  # \（2,）数组作为index.axis=1,处理一行，上下叠加
print(m.shape)
print(n.shape)
print(m)
