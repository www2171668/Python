""""""
import numpy as np

# %% [ ] 索引和切片  ①、索引是找指定位置的数据；切片是用冒号找范围位置的数据  ②、索引会降维，而切片不回改变维度
arr03 = np.arange(1, 25).reshape((2, 3, 4))
print(arr03)

# 1、索引
print(arr03[:, :, 0])  # * 注意排列形式的转变。降维处理
print(arr03[0, :, :])

# 2、切片
print(arr03[:, :, 0:1])  # * 前包后不包。排列形式不变
print(arr03[:, 0:1, 0:3:2])

# %% 索引和切片获取的元素为浅拷贝
arr02 = np.arange(1, 13).reshape(3, 4)
arrs02 = arr02[1:2, :]
arrs02[:] = 1001
print(arr02)

# %% 布尔索引 ★
arr = np.random.random((4, 3))
print(arr)

arr2 = arr < 0.5  # \ 返回的是一个bool矩阵
print(arr2)

arr3 = arr[arr2]  # * 返回True对应的元素。输出矩阵形状改变
print(arr3)

# %% np.newaxis：插入新维度
a = np.array([1, 2])
aa = a[:, np.newaxis]  # 2行1列
bb = a[np.newaxis, :]  # 1行2列
print(aa.shape, bb.shape)
print(aa, bb)

a = np.array([[1, 2], [3, 4]])
aa = a[np.newaxis, :, :]  # 1快 2行 2列
print(aa.shape)
print(aa)

# %% 例题
names = np.array(['joe', 'susan', 'tom'])
scores = np.array([
    [98, 86, 88, 90],
    [70, 86, 90, 99],
    [82, 88, 89, 86]
])
classes = np.array(['语文', '数学', '英语', '科学'])

print('joe的成绩是:', scores[names == 'joe'])  # 返回 [scores[[True,False,False]] ；可以看成花式索引，True和False都是索引值
# 由于只有一个[]，就是对行进行操作,相当于print('joe的成绩是:',scores[[True,False,False],:])

print('joe的成绩是:', scores[[True, False, False], :])  # print(scores[[0],:])   由于使用的是单行花式索引，会造成多一个[ ]的情况。
# 故使用花式索引取单行数据时，要注意加reshape

print('joe的成绩是:', scores[names == 'joe'].reshape(-1))  # reshape(-1) 自动匹配行列数 ；在此将高（二）维数组转换为低（一）维数组

print(names == 'joe', classes == '数学')

print('joe的数学成绩是：', scores[names == 'joe', classes == '数学'])  # 按照索引规则，names是行索引，classes是列索引 —— 推荐写法

# %% 以下均为不常用  ————————————————————————————————————————————————
print(1)

# %% 花式索引 ★  [ [ ] , [ ] ] 多个索引的组合形式
arr02 = np.arange(1, 13).reshape(3, 4)
print(arr02)
print(arr02[[0, 2], :])  # 获取第0行和第2行数据 ；-》也可以写成 arr[[0,3]]

# 索引与花式索引的区别：
print(arr02[:, 0])  # 有降维
print(arr02[:, [0]])

print(arr02[0, :])  # 有降维
print(arr02[[0], :])  # 获取第0行数据，一般不会这么用；-》由于是单行花式索引，会造成输出多一个[]的情况，一般可以用reshape(-1)来自动修正[]数量

print(arr02[[0, 1, 2], [1]])  # 花式索引
print(arr02[[0, 1, 2], [1, 1, 1]])      # print(arr02[[0,1,2],[1,2]])   # 会报错


# %% 索引器  arr [ np.ix_(*) ] 使用索引器来获取范围位置内的数据  不常用
arr02 = np.arange(1, 13).reshape(3, 4)
num = arr02[np.ix_([0, 1, 2], [0, 1])]  # \ 查找a00,a01,...等元素组成局部矩阵
print(num)

# %% 深拷贝 与 花式索引        花式索引返回的始终是数据的副本,修改数据不会改变原数组的数据
arrs02 = arr02[[0, 1, 2], [3]]  # 花式索引，深拷贝
arrs02[:] = 50
print(arr02)
