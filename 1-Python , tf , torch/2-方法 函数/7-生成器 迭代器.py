""""""

# %% 生成器    函数中存在yield时，函数将变成一个generator迭代器
# 使用生成器可以达到延迟操作的效果，即在需要的时候产生结果而不是立即产生就结果，节省资源消耗。和声明一个序列不同的是，生成器在不使用的时候几乎是不占内存的。

def getNum(n):
    i = 0
    while i <= n:
        print(i)  # 打印i
        return i  # 返回一个i ,结束函数的运行
        yield i  # 将函数变成一个generator
        i += 1

print(getNum(5))

a = getNum(5)
# 使用生成器 通过 next()方法
print(next(a))  # 输出yield返回的值
print(next(a))
print(next(a))
print(next(a))
print(next(a))
print(next(a))

# for循环遍历一个生成器
for i in a:
    print(i)

a = [x for x in range(10000000)]  # 这样生成一个很多数据的列表会占用很大的内存
print(a)
a = (x for x in range(10000000))  # 不是元组推导式，（）是一个生成器
print(a)
print(next(a))
print(next(a))
print(next(a))

# %% send() 将()中的值发送到 上一次yield的地方，并继续运行yield

def gen():
    i = 0
    while i < 5:
        temp = yield i  # 是赋值操作吗？不是
        # 使用了yield之后，该gen函数自动转化为了一个生成器
        print(temp)  # 因为 yield 之后返回结果到调用者的地方，暂停运行 ，赋值操作没有运行
        i += 1

a = gen()
print(next(a))
print(next(a))
print(a.send('我是a'))

# %% 可迭代对象Iterable：list string dict （int不是）
# 迭代器iterator：可以被next()函数调用的，并不断返回下一个值的对象.      iter() 将可迭代对象变成迭代器     next() 输出当前元素，并停止

list01 = [1, 2, 3, 4, 5]  # \ 可迭代对象
for i in list01:
    print(i)

a = iter(list01)
print(a)
print(next(a))
print(next(a))
print(next(a))

def f():
    print('start')  # 在 a=f()的时候，并没有运行方法内的表达式；到next(a)时，开始运行
    a = yield 1  # 表达式都是从右向左开始,首先返回1 然后a=1 yield 1  a=1
    print(a)
    print('middle....')
    b = yield 2  # 2这个值只是迭代值，调用next时候返回的值
    print(b)
    print('next')
    c = yield 3
    print(c)  # 未运行

a = f()
print(next(a))
print(a.send('msg'))  # 将msg传递给a= 右边，并继续运行方法内的表达式
print(a.send('msg1'))  # 在将msg1传入职前，优先运行yield，send默认具有next的功能，接收yield的值
