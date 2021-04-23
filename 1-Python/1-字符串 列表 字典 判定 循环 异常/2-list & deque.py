# %% 访问、修改列表元素
# 1、通过下标查找
list01 = ['jack', 'jane', 'joe', 'black']
print(list01[2])

# 1、通过元素名查找元素
list02 = ['jack', 'jane', 'joe', 'black']
name = 'jack'
print(name in list02)

# 2、修改列表元素
list01 = ['jack', 'jane', ['leonaldo', 'joe'], 'black']
list01[0] = 'lili'

# %% 增加列表元素   .append 往列表末尾增加元素     .insert 往列表指定位置添加元素 （位置，元素）     .zip()合并列表
# 1、append
list02 = ['jack', 'jane', 'joe', 'black']
list02.append('susan')
print(list02)

# 2、insert
list02 = ['jack', 'jane', 'joe', 'black']
list02.insert(1, 'susan')
print(list02)

# 3、列表元组
list03 = ['a', 'b', 'c']
list04 = [1, 2, 3]
print(list(zip(list03, list04)))  # \ [('a', 1), ('b', 2), ('c', 3)]

# %% 删除列表元素   .pop 通过下标删除元素，默认删除最后一个    .remove 通过值删除元素
# 1、pop
list03 = ['jack', 'jane', 'joe', 'black']
print(list03.pop(1))  # 执行删除操作 并且返回删除的元素
print(list03)

# 2、remove
list03 = ['jack', 'jane', 'joe', 'black']
list03.remove('jane')  # 通过元素的值进行删除
print(list03)

# %% enumerate(*) 返回的是一个enumerate对象  ★★
# * 对于一个可迭代的（iterable）/可遍历的对象（如列表、元组、字符串），enumerate将其组成一个索引序列（从0开始）
# * 该方法常用于在for循环中，用k,v同时获得索引和值
list05 = ['joe', 'susan', 'black']
print(list(enumerate(list05)))  # * 输出列表元祖 [(0, 'joe'), (1, 'susan'), (2, 'black')]。 直接输出enumerate(list05)是无意义的

dict01 = {k: v for k, v in enumerate(list05)}  # \ k 依次获取索引值，最后转为字典。
print(dict01)  # 输出 {0: 'joe', 1: 'susan', 2: 'black'}

# %% 将列表用for循环添加到另一个字典中    *.update({*})  添加字典元素，配合{key:value}使用
names = ['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe', 'Alice', 'Sherry', 'Eva']
name = {k: v for k, v in enumerate(names)}

dict01 = {}
for k, v in enumerate(names):
    dict01.update({k: v})
print(dict01)

# %% deque(* ,maxlen)：一种两端都可以进行操作的list。
#  .append() 或 .extend()：在右边添加元素      .extendleft()：在左边添加元素
#  .pop()：删除右边元素右       .popleft()：删除左边元素

from collections import deque

d = deque()  # \ deque列表元素，会有一个前缀。
d.append('1')  # \ 右加
print(d)

d = deque('25843')  # \ 自动切割str
print(d)

d = deque(maxlen=2)  # maxlen用于限制deque的长度。当限制长度的deque增加超过限制数的项时, 另一边的项会自动删除:
d.append(1)
d.append(2)
d.append(3)
print(d)  # deque([2, 3], maxlen=2)


