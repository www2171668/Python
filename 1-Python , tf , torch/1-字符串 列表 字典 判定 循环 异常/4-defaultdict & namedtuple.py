""""""
# %% defaultdict   带默认值的字典（访问dict字典不存在的元素时，会报错显示keyerror）
"""
    defaultdict()在dict()的基础上添加了missing(key)方法：在调用一个不存在的key时，调用“missing”，返回一个int,set,list,dict对应的默认数值，不会出现keyerror的情况  ★
    defaultdict(list)保留了list的方法
"""
from collections import defaultdict

a = {}
print(a["a"])  # \ >>>KeyError,"a"

b = defaultdict(int)
print(b["a"])  # \ 以0填充缺失数据

c = defaultdict(list)
print(c["a"])  # \ 以无填充缺失数据

c = defaultdict(list)
c["a"].append("c")
print(c)  # \ 输出：defaultdict(<class 'list'>, {'a': ['c']})

# %% 案例一：int型defaultdict()
s = 'mississippi'
d = defaultdict(int)
for i in s:
    d[i] += 1
print(d)  # \ 输出：defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})

# %%  namedtuple（类名，类属性）:命名元组对象，常用于构建只有少量属性，没有方法的类

from collections import namedtuple

Student = namedtuple('Student', ['name', 'sex', 'age'])  # 类的声明
xiaoming = Student('小明', 'man', '20')  # 类的实例化

print(xiaoming.name)  # \ 小明
print(xiaoming._fields)  # ._fields :查看所有属性:('name', 'sex', 'age')

# -》._replace()：修改对象属性
xiaoming = xiaoming._replace(sex='unknow')
print(xiaoming)

# -》._asdict()：将对象转换成字典
xiaoming._asdict()
print(xiaoming)  # \ OrderedDict([('name', '小明'), ('sex', 'unknow'), ('age', '20')])
