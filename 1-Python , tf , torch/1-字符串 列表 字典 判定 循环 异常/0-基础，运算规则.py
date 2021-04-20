# %% 输出方式 .format() 或  %()    %s 输出字符串    %d 输出数字       end     \n
print('a=',10)

print('a={},b={}'.format(10, 'y'))
print('大家好我叫%s,我今年%d岁，我来自%s' % ('joe', 18, '上海'))

print('1', end='')  # * 为end传递一个空字符串，这样print函数不会在字符串末尾添加一个换行符，而是添加一个空字符串
print('5')  # 5会接在1后面

print("result:\n{}".format(10)) # * 回车

# %% TRUE 为1   FALSE 为0
print(int(True))

# %% 拷贝类型
# * 浅拷贝，直接赋值，共享内存空间
# * 深拷贝，copy.deepcopy 深拷贝赋值   import copy
import copy

a = [1, 2, 3]
b = copy.deepcopy(a)

# %% 矩阵加法：同列维的矩阵才能相加；否则会被认为增加维度
# \ np.transpose(*) 或 *.transpose() 转置。  list 没有 .T 方法，DataFrame可以用 .T
import numpy as np

list1 = [[1, 1, 1], [2, 2, 2]]
list2 = [[1], [1]]
print(list1 + list2)

list3 = np.transpose(list1)
list4 = np.transpose(list2)
print(list3 + list4)

arr1 = np.array([[1], [2]])
arr2 = np.array([[1], [1]])
print(arr1 + arr2)

# %% np.dot() 矩阵乘法
list1 = [[1, 1, 1], [2, 2, 2]]
list2 = [[1], [1], [2]]
print(np.dot(list1, list2))
print(np.dot(list1, 2))  # \ 不可以用 list1*3 （扩张）
#%% 整数判断    isinstance(*,类型)
if not isinstance(10, int):
    raise ValueError('分数必须是整数才行')