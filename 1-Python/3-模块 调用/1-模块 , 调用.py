""""""
#%% from 模块名 import 函数名 as 重命名        from 模块名 import *

# 1、导入指定函数
from time import sleep as s
print('start')
s(5)
print('stop')

# 2、导入所有函数
from math import *
print(ceil(1.1))  #向上取整

# %% 调用方法
from function import words  # 引用函数

words("调用函数", 1)

# %% 调用模块
from function import Restaurant

R1 = Restaurant('大酒店')
R1.describe_restaurant('紫菜')
