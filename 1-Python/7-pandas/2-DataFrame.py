""""""
import pandas as pd
import numpy as np

# %% 1、通过二维数组创建 DataFrame(data, index, columns, dtype)  index 索引 ; columns 列名     .values 获取DataFrame的元素，转换为array
arr = np.array([
    ['joe', 70],
    ['suan', 80],
    ['black', 90]
])
df02 = pd.DataFrame(arr)
print(df02)
print(df02.values)

df02 = pd.DataFrame(arr, index=['one', 'two'], columns=['name', 'score'])  # 使用columns 添加列名，注意数量必须和列数对应
print(df02)

# %% 例题
def parseRecord(record):  # 函数功能：将column2中的值转为标签类型（从0开始的序列）
    result = []
    #     print(record)
    columns = ['column1', 'column2']
    for column, v in zip(columns, record):  # 多参循环
        if column == 'column2':
            if v == 70:
                result.append(0)
            elif v == 80:
                result.append(1)
            elif v == 90:
                result.append(2)
            else:
                result.append(np.nan)  # 转为 NaN 一会可以删除无效样本
        else:
            result.append(v)  # 原数不动
    print()
    result = pd.Series(data=result, index=columns)
    print(result)
    return result  # return返回Series类型的数据

# %% ﹡.apply(func[, args[, kwargs]]) ：返回func()值；通常使用﹡.apply(lambda i: 函数(i), axis) ★★★
arr = np.array([
    [1, 70],
    [2, 80],
    [3, 90]
])
column = ['column1', 'column2']
df02 = pd.DataFrame(arr, index=['one', 'two', 'three'], columns=column)
print(df02)

# 依次遍历df02的行，每一行的数据赋予r，传入函数；apply转()内的返回值为DataFrame，赋予datas  ★★★
# 可以理解为，仅仅是对df02每一行进行了处理（处理为Series）
datas = df02.apply(lambda r: parseRecord(r), axis=1)  # axis=1跨列处理：因为lambda中的是Series数据，对每一列进行转置处理，一行一行叠加
# datas = df02.apply(lambda x: pd.Series(data = parseRecord(x)), axis=1)
print(datas)

# %%  2、通过字典创建      在Series中，字典中的key是列表中的行索引（index）；在Dataframe中，字典中的key是表格中的列名（columns）
dict01 = {
    'name': ['joe', 'susan', 'black'],
    'age': [19, 19, 20],
    'sex': ['men', 'women', 'men'],
    'classid': 2  # 非匹配列，自动填充整个列
}
df03 = pd.DataFrame(dict01, index=['one', 'two', 'three'])
print(df03)

# %%  DataFrame的属性 1、获取DataFrame数据 - 列操作 —— DataF名[]     —— ﹡[索引名]   或 ﹡.columns[索引号]    获取行数据，通常使用索引号
columns = ['name', 'age', 'sex']
print(df03[columns[0:2]])

print(df03[['name', 'age']])  # 注意返回的是DataFrame

print(df03['name'])  # 没有
# print(df03['name':])  # 不能通过列名用切片 —— 得Series。而行名可以 ！！！！！！
print()

print(df03.columns[2:])  # 通过编号使用切片，注意返回的是Index
print()
print(df03[df03.columns[2:]])

# %%
list01 = [[1, 2, 3], [2, 3, 4]]  # 列表
print(list01[0])

array01 = np.array([[1, 2, 3], [2, 3, 4]])  # 数组
print(array01[0])

# %%  2、增加、修改、删除DataFrame数据 - 列操作       ——.drop(﹡, n)  ：n = 0时删除行，n = 1时删除列 ★       ——.pop() 或 del () 删除列
# Series和DataFrame在修改数据时有一个很大的区别，[]中若使用数字，Series只能修改已有数据，DataFrame可以增加列（行），故推荐使用列名（行名）来进行相应操作

# 1、列增加
df03['address'] = ['北京', '上海', '广州']  # 若原表中没有[]中的columns值，则在末尾加列
# df03[4] = ['北京','上海','广州']

# 2、列修改
df03['classid'] = [1, 2, 3]  # 若原表中有[]中的columns值，则对该列完全修改
# df03['classid','two'] = 50   # 无法对某一个元素进行修改

# 3.1、列删除
del (df03['address'])  # df03.pop('address')

# 3.2、列删除
df03 = df03.drop('name', 1)

# %%  3、获取DataFrame数据 - 行操作     ——.loc[]    使用索引名称检索 /.iloc[]       使用索引号检索

print(df03.loc['one'])  # 获取one行
# print(df03[1:])   #一般不这么用
print()
print(df03.loc['one', 'name'])  # 获取one行，name列
print()
print(df03.loc['one':'three', 'name'])  # 获取one-three行，name列 ，字典型跨距索引
print()
print(df03.iloc[1:2, 1])  # 使用数字索引（即行号）时，使用iloc

# %%  4、增加、修改、删除DataFrame数据 - 行操作

# 1、行增加
df03.loc['four'] = [12, 1, 'jack', 'men']  # 若原表中没有[]中的index值，则在末尾加行

# 2、行修改
df03.loc['four'] = [15, 1, 'jacks', 'men']  # 若原表中有[]中的inxed值，则对该行进行完全变更；此时必须保证列数量上对齐
df03.loc['one', 'classid'] = 3  # 修改单个元素

# 3、行删除
df03 = df03.drop('four')

