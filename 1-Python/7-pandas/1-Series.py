""""""
import numpy as np
import pandas as pd

# %% Series的创建
ser01 = pd.Series()  # 创建一个空的Series
print(ser01)

# %% 1、通过 数组array 创建Serise     pd.Series(data,index,dtype)   data 数据     index 索引,轴标签
data = np.array(['a', 2, np.NaN, 10.0])
ser02 = pd.Series(data)
print(ser02)  # 默认以从0开始的序列作为索引

ser03 = pd.Series(data=['a', 2, np.NaN, 10.0])
print(ser03)

ser04 = pd.Series(data=[70, 80, 90], index=['语文', '数学', '英语'])  # 可以加上 dtype = np.float64
print(ser04)

# %% 2、通过字典创建Series，字典的key是Series列中的index
data = {'a': 1, 'b': 2, 'c': 3}
ser05 = pd.Series(data)
print(ser05)

# %% Series的属性       —— .dtype / .index / .values  查看series的元素类型 / 索引 / 值
ser02 = pd.Series(data=['a', 2, np.NaN, 10.0])
print(ser02.dtype)  # 只有在元素全是int的情况下，dtype才会默认为int，否则都被默认转为字符串类型
print(ser02.index)
print(ser02.values)

# %%1、获取series数据     —— Ser名[ 索引号 ] / Ser名[ 索引名 ]        —— 字典式索引
ser06 = pd.Series(data=[70, 80, 90], index=['语文', '数学', '英语'])
print(ser06)

# 1、通过索引方式获取
print(ser06['语文'])

# 2、通过索引方式获取
print(ser06[0])  # 由于索引列不属于数据，所以不能使用ser06[0,:]，不要被索引列迷惑了
print(ser06[:2])  # * 前包后不包
print(ser06['语文':'英语'])  # * 前后都包，字典型跨距索引属于特殊情况

# %% 2、增加、修改Series数据
ser06['地理'] = 100  # 增加数据
ser06[2] = 50  # 修改数据
print(ser06)

# %% 3、缺失值检测  ★      通过index进行检测，用isnull进行bool判定，获得bool矩阵后检索相应缺失值        —— pd.isnull(*) 返回一个区分“NaN(不是一个数字)”的布尔类型Series。和np.isnull一致
scores = pd.Series({'tom': 70, 'joe': 80, 'susan': 90})
new_index = ['tom', 'joe', 'susan', 'anne']  # 增加了一个缺失值 - 只有索引没有值

scores = pd.Series(scores, index=new_index)  # 1、检测缺失值
print(scores)

# 获取缺省值 + 过滤出缺失值
boolnull = pd.isnull(scores)  # 或 notnull
print(boolnull)
print(scores[boolnull])

# %% Series的标题       —— .name Series标题       —— .index.name 索引标题
scores = pd.Series({'tom': 70, 'joe': 80, 'susan': 90})
scores.name = '考试成绩'
scores.index.name = '科目'

# —— .head(*) 获取前5组数据
# —— .tail(*) 获取后5组数据
print(scores.head(2))  # （2）获取前两组数据

print(scores.tail(2))  # （2）获取前两组数据

# %% Series的运算 + 逻辑 + 聚合     Series可以使用numpy的属性和方法
series = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(series)
print(series + 10)
print(series[series > 1])
print(np.max(series))

# %% Series与Series相加
series1 = pd.Series({'a': 1, 'b': 2, 'c': 3})
series2 = pd.Series({'a': 10, 'b': 20, 'c': 30})
print(series1 + series2)

series1 = pd.Series({'a': 981, 'b': 211, 'c': 9527, 'd': 100})
series2 = pd.Series({'a': 10, 'b': 20, 'c': 30, 'e': 200})
print(series1 + series2)  # 无交集的将会丧失原值
