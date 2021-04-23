# %% 访问、修改字典元素    通过key访问、修改
# 1、访问字典
dict02 = {'name': 'joe', 'age': 18, 'address': '上海'}
print(dict02['name'])  # \ print(dict[1]) 调用是错误的

# 2、修改字典元素 通过指定key修改value
dict02['name'] = 'jack'
print(dict02)

# %% 增加字典元素    通过key增加字典    *.update({*}) 增加字典元素
# 增加字典元素 字典中不存在该key     字典可以嵌套 dict03['1001'] = {'name':'小张','sex':'男'}
dict02['hobby'] = '足球'
print(dict02)

dict02.update({'Height': 170})
print(dict02)

# %% 删除字典元素   *.pop(*) 通过key删除指定位置元素       *.clear() 清空字典中的元素
dict02.pop('hobby')
print(dict02)

dict02.clear()  # 清空字典中的元素
print(dict02)

# %% 字典函数    *.items()   ※
#   *.keys() 输出所有key值      *.values() 输出所有value值   *.get(*) 通过key获取value。若无该key，则输出None  ★
#   *.items() 将字典转为items型的列表元组[( ),( ),( )]；需要继续用list将items型转列表型 ★★★   常用于拆分字典元素，通过for循环遍历字典中的key和value
#   iteritems(*) 和items功能相同，也需要转list   需要from six import iteritems

dict02 = {'name': 'joe', 'age': '18'}

# 1、分别输出key和value
print(dict02.keys())
print(dict02.values())
print(dict02.get('name'))

# 2、同时输出key和value
print(dict02.items())
print(list(dict02.items()))  # \ 转为列表元祖 [('name', 'joe'), ('age', '18')]
for k, v in dict02.items():  # \ k获取索引，v获取值
    print(k, v)

from six import iteritems

print(iteritems(dict02))
print(list(iteritems(dict02)))  # 转为列表元祖 [('name', 'joe'), ('age', '18')]
for k, v in iteritems(dict02):  # k获取索引，v获取值
    print(k, v)

# %% 字典排序    sorted(*) 接收字典的key，对key进行排序，返回列表     sorted(*.items()) 接收字典的key和value，通过key进行整体排序，返回列表元组
dic = {'a': 1, 'c': 2, 'b': 3}
dic01 = sorted(dic)  # \ 只接收了dic字典的key，并进行排序
print(dic01)  # \ 输出['a', 'b', 'c']

dic01 = sorted(dic.items())
print(dic01)  # \ 输出[('a', 1), ('b', 3), ('c', 2)]

# key的应用
dic01 = sorted(dic.items(), key=lambda x: x[1])  # * x 传入 dic.items()，然后返回x[1]，即原字典的value值；再通过sorted，按照x[1]的值来排序
print(dic01)  # \ 输出 [('a', 1), ('c', 2), ('b', 3)]

