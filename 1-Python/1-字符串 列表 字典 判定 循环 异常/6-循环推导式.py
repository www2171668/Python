""""""
import numpy as np

# %% 循环推导式     (列表元素 for 变量 in 迭代对象)
list03 = [1, 2, 3, 4]
num = sum(i + 0 for i in list03)
print(num)

# %% 列表推导式     [列表元素 for 变量 in 迭代对象 判断语句/处理语句]   注意[] ★★★
list03 = []
for i in range(3, 10):
    if i % 2 == 0:
        list03.append(i)
print(list03)

list03 = [i for i in range(3, 10) if i % 2 == 0]
print(list03)

p = [np.random.random() for x in range(6)]  # \ 随机列表
print(p)

# %% 嵌套列表推导
# * [列表元素 for i in 迭代对象1 for j in 迭代对象2  判断语句/处理语句]  按从左到右的顺序执行嵌套的for循环 ★
# * [[列表元素 for i in 迭代对象1] for j in 迭代对象2]   从外[]到内[]

# 1、
names = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Joe'], ['Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Eva']]

list04 = []
for i in names:
    for j in i:
        if len(j) > 4:
            list04.append(j)
print(list04)

lsit04 = [j for i in names for j in i if len(j) > 4]
print(lsit04)

# 2、
result = [[1, 2], [1, 2], [1, 3]]
result = [[x[i] for x in result] for i in range(2)]  # 限制性外[]，再执行内[]
print(result)
