""""""
import numpy as np

# %% 判断 & 循环 & 跳转
# \ while 为True时进入
# while: ... else:      for、while循环都有一个可选的else分支，只有在循环迭代 正常完成 之后才会执行。

# * break 终止（跳出）当前内圈循环     continue 结束当前内圈循环，直接开始下一次循环

# %% list和np的bool    list只会输出true和false / np会输出每一个数字的判断
list01 = [1, 0, 0, 0, 1]
array01 = np.array([1, 0, 0, 0, 1])

condition = (array01 == 1)
print(condition)

# %% for 变量 in 迭代对象：     迭代对象Iterable包括 list string dict （int不是）     -》pass 仅提示后面没有语句了，没有实际意思
list01 = ['joe', 'susan', 'jack', 'Tom']
for i in list01:
    print(i)
    pass

# \ 嵌套结构
list02 = [[1, 1], [2, 2]]
for i in list02:
    print(i)

list04 = [(1, 2), (3, 4), (5, 6)]
for id1, id2 in list04:
    print(id1)

# %% zip(list1,list2) 多值循环时使用 ★
list01 = ['joe', 'susan', 'jack', 'Tom']
list02 = ['1', '2', '3', '4']
for x, y in zip(list01, list02):  # \ 必须加zip，否则too many values to unpack (expected 2)
    print(x, y)

dict01 = {x: y for x, y in zip(list01, list02)}  # \ 循环推导式
print(dict01)
