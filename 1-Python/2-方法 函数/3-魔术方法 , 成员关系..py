""""""

# 魔法方法
'''
    每个魔法方法都对应了一个Python的内置方法或操作，比如__str__对应str方法，__lt__对应小于号<等。
    使用魔法方法可以使Python的自由度变得更高
    -》触发时机:当对象被内存回收的时候自动触发[1.页面执行完毕回收所有变量 2.对象的所有变量被del的时候]
    -》功能:处理对象使用完毕时候的成员或者资源的回收
    -》参数:self
    -》返回值:无
'''


class Human:
    money = 1000
    age = 18
    name = '小松松'

    # del魔术方法  析构方法
    def __del__(self):
        print('del方法被触发')


a = Human()
del a  # \ 调用魔法方法


# %% __call__ ：魔法方法的使用   无需 类名.方法名 调用
class A:
    def __call__(self, param):
        print('这是一个函数')
        print('传入参数值为：{}'.format(param))

        res = self.forward(param)
        return res + 1

    def forward(self, input_):
        print('forward 方法被调用了')
        print('in  forward, 传入参数值为: {}'.format(input_ + 1))
        return input_ + 1  # 执行到此返回2给res，然后+1作为类调用魔法方法的返回值


a = A()
output_param = a(1)  # \ 调用魔法方法，无需 类名.方法名 调用
print("对象a输出的参数是：", output_param)


# %% __bool__和__len__

class A(object):
    pass


class B(object):
    def __bool__(self):
        print("__bool__")
        return True


class C(object):
    def __len__(self):
        print("__len__")
        return 0


b = B()
c = C()
print(bool(b))  # True
print(bool(c))  # False


# %% __contains__优于__iter__优于__getitem__方法
# __contains__方法把成员关系定义为对一个映射应用键，以及用于序列的搜索

class Iters:
    def __init__(self, value):
        self.data = value

    def __getitem__(self, i):
        print('get[%s]:' % i, end=' ')
        return self.data[i]

    def __iter__(self):
        print('iter=>', end=' ')
        self.ix = 0
        return self

    def __next__(self):
        print('next:', end=' ')
        if self.ix == len(self.data):
            raise StopIteration
        item = self.data[self.ix]
        self.ix += 1
        return item

    def __contains__(self, x):  # in 和not in 操作的时候会自动调用这个函数
        print('contains: ', end=' ')
        return x in self.data


x = Iters([1, 2, 3, 4, 5])
print(3 in x)  # contains:  True
for i in x:
    print(i, end=', ')  # iter=> next: 1, next: 2, next: 3, next: 4, next: 5, next:
print()
print([i ** 2 for i in x])  # iter=> next: next: next: next: next: next: [1, 4, 9, 16, 25]
print(list(map(bin, x)))  # iter=> next: next: next: next: next: next: ['0b1', '0b10', '0b11', '0b100', '0b101']
i = iter(x)
while True:
    try:
        print(next(i), end=' @ ')  # iter=> next: 1 @ next: 2 @ next: 3 @ next: 4 @ next: 5 @ next:
    except StopIteration:
        break
print()
print(x[0])  # get[0]: 1   __getitem__支持索引
print(x[:-1])  # get[slice(None, -1, None)]: [1, 2, 3, 4]
