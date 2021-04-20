""""""

# %% 匿名函数  lambda [参数]: 表达式     返回表达式的值 ★

# 1、无参匿名函数
s = lambda: '哈哈啊哈'  # 通过；lambda 声明一个匿名函数 并且 赋值给 s
print(s())  # 通过s()调用匿名函数

# 2、有参匿名函数
s = lambda x: x * 2  # 将3传到：左边的x上；x*2为返回值
print(s(3))  # 返回6

# 3、多参匿名函数
s = lambda x, y: x + y
print(s(3, 4))  # 返回7

# 4、矢量化三元运算 匿名函数
s = lambda x, y: x if x > 2 else y
print(s(3, 2))  # 返回3

# %% 高阶函数        map     filter      reduce         map和filter都需要list()转换
# *    -》map(lambda i:表达式,迭代对象)：返回map函数，需要转为list才可获得列表推导式 - 返回由表达式值组成的列表
# *    -》filter(lambda i:判断条件,迭代对象)：返回filter，需要转为list才可获得列表推导式 - 返回满足判断条件的迭代元素组成的列表
# *    -》reduce(lambda x,y:表达式,迭代对象,x=初始值)：直接返回表达式值。赋予x一个初始值，y接收迭代对象元素，表达式计算后，新值赋予x，以此循环
# 1、map
list01 = [1, 3, 5, 7, 9]
list02 = [2, 4, 6, 8, 10]
new_list01 = [x * 2 for x in list01]  # 循环推导式
new_list02 = map(lambda x: x * 2, list01)
new_list03 = map(lambda x, y: x * y, list01, list02)
print(list(new_list02))

# 2、filter
list02 = [2, 4, 6, 8, 10]
new_list01 = filter(lambda x: x > 4, list02)  # \ [6, 8, 10]
new_list02 = map(lambda x: x > 4, list02)  # \ [False, False, True, True, True]
new_list03 = [i for i in list02 if i > 4]
print(new_list03)

# 3、reduce
from functools import reduce

list02 = [2, 4, 6, 8, 10]
new_list = reduce(lambda x, y: x + y, list02, 0)  # \ 0是x的初始值，y接收遍历值；每一次运算后，x+y的值都会给到x
print(new_list)

# %% 循环推导式练习
name = ['joe', 'susan', 'black', 'lili']
age = [18, 19, 20, 21]
sex = ['m', 'w', 'm', 'w']

# 将用户英文名、年龄、性别三个集合的数据结合到一起，形成一个元祖列表
new_user01 = [(i, j, k) for i, j, k in zip(name, age, sex)]
new_user02 = list(map(lambda x, y, z: (x, y, z), name, age, sex))

# 过滤性别为男的用户
man_user = [i for i in new_user01 if i[2] == 'm']
print(man_user)
