# %% type(*) 查看数据类型     格式转换  str(*)  列表元组的转换  list(*)  tuple(*)
dict02 = {'name': 'joe', 'age': 18, 'address': '上海'}
str1 = str(dict02)
print(str1)
print(type(str1))

# %% 列表[ ]  元组( )  字典{ : , :}
list01 = ['a', 'b', 1, 2]
tup01 = ('a', 'b', 1, 2)  # * 具有不可更改性  如 tup01[0] = 9   #不支持更改
dict01 = {'name': 'joe', 'age': 19, 'address': '上海'}

# %% 字符串常用函数
"""
    -》*.find(*) 返回字符所在位置，从0开始计数。如果没有该字符，则返回-1
    -》*.index(*) 返回字符所在位置，从0开始计数。如果没有该字符，则报错
    -》*.count(*) 返回字符所在位置，从1开始计数，可以在指定区域内进行查找

    -》len(*)  或 .len（）

    -》+  拼接字符串
    -》*.replace(*,*)  后替前   ※
    -》*.split(*) 按照()中的内容拆分字符串，返回拆分后元素组成新的列表   ★
"""
str01 = 'i love python o'

print(str01.find('l'))  #z 1、find   2 空格也算
print(str01.index('o'))  # 2、index   3
print(str01.count('i'))  # 3、count   1

# 1、+
str01 = '2' + '1' + '3ccc'
print(str01)

# 2、replace
str01 = 'i love python o'
print(str01.replace('p', 'PPPP'))

# 3、split
print(str01.split(' '))

# %% 截取字符串,用 [] 调用
str1 = 'abcdfeg'
print(str1[2:])
print(str1[:2])

print(str1[2:6])  # \ 前包后不包
print(str1[1:5:2])  # \ 是步长
print(str1[-2:-6:-1])  # \-1步长
