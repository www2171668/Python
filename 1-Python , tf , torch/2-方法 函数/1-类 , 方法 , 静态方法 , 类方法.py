""""""

# %%  class 类名():    构造方法  def __init__(self,*):     def 方法名(self,*):
# 属性(变量) - 函数外的属性为全局属性(类属性) - 函数内的属性是局部属性（实例属性）
# 局部变量：函数中声明的变量，只能在该函数中使用   全局变量：外部声明的变量，共享内存地址。   global 在函数中强行声明局部变量

"""
    -》在构造函数外声明的属性是类属性
       类属性可以用 类名.属性名 对象名.属性名 两种方式访问
       ①、内部访问时使用 类名.属性名 访问
       ②、外部访问时使用 对象名.属性名 访问
"""

# def __init__(self,变量名): 重写构造函数。构造方法在实例化的时候会自动调用
"""
   self不是关键字，代表的是当前对象，类似于this.  方法()中的变量是形参
   ①、通过self.声明的变量是实例属性，内部访问时，通过 self.属性名 访问，外部访问时通过 对象名.属性名 访问
   ②、加了__ 下划线的是私有属性，如果外部要访问或修改私有属性，需要预留一个接口 ，即一个带返回值的方法
"""

# def 方法名(self)： 外部访问时使用 对象名.方法名 访问

# 全局变量
globaltext = 100


class person:
    country = '中国'  # 全局属性

    def __init__(self, name, age, sex, address):  # 类的构造函数，构造函数中的属性是局部属性
        print('我是构造方法，在实例化得时候调用')
        self.name = name
        self.age = age
        self.sex = sex
        self.__address = address  # 双下划线开头的属性是私有属性

    def getName(self):
        print('我叫：%s,我来自%s,我住在%s,学号是%d' % (self.name, person.country, self.__address, globaltext))  # 全局变量可以直接使用
        # 使用类属性时，最好用 类名.属性名 来使用。但直接用 self.属性名 也可以  ★

    def getNum(self):
        return 1        # * return   返回值，执行到return时，后面的代码都不会执行

    def getAddre(self):
        return self.__address  # 私有属性输出方法


# 1、实例化对象
people01 = person('joe', 19, '男', '上海')

# 2、访问实例属性 & 类属性 & 私有属性
print(people01.name)  # 通过 对象名.属性名 访问实例属性(对象属性)    print(people01.__address) 外部无法使用 对象名.属性名 方式访问私有属性
print(people01.country)  # \ 或 print(person.country)
print(people01.getAddre())
print(people01.getNum())

# 3、调用实例方法
people01.getName()  # 通过 对象名.方法名

# %% 内置类属性    .__dict__  将实例对象的属性和值通过字典的形式返回
print(people01.__dict__)

# %%  静态方法 @staticmethod     类方法 @classmethod
'''
    -》@staticmethod 在该关键词下声明的方法为静态方法，静态方法不需要传递参数，所以()里面也没有self
        静态方法不能访问实例属性 如self.name
        静态方法只能访问类属性 如person.country

    静态方法和类方法可以使用 类名.方法名 调用，一般不用 对象名.方法名
'''


class person:
    country = '中国'

    def __init__(self, name, age, sex, country):
        print('我是构造方法，在实例化得时候调用')
        self.name = name
        self.age = age
        self.sex = sex
        person.country = country

    @staticmethod
    def aa():
        print('我来自：%s' % person.country)  # \ 静态方法中只能使用类属性


people01 = person('joe', 19, '男', '美国')
person.aa()  # \ 或 people01.aa()


# %% 装饰器    @property 把类里的函数直接对外，以变量的形式进行表述，使得函数参数的给予变成赋值的形式。与之相关的存在以下两种工具：
# ①、@<函数名>setter 装饰器，使函数以赋值的形式对外索取参数。
# ②、@<函数名>deleter，可以直接用del变量名的方式删除变量。

class Student:
    def get_score(self):
        return self._score

    def set_score(self, value):
        self._score = value
        print(value)


s = Student()
s.set_score(9999)


class Student2:
    @property
    def score(self):
        return self._score

    @score.setter  # score装饰器的属性
    def score(self, value):
        self._score = value
        print(value)


# 加上@property装饰器，可以将get_score()方法变为属性
# 根据@property创建附属装饰器@score.setter,负责把set_score()方法变成属性赋值,这样处理可以使调用既可控又方便

s = Student2()
s.score = 60  # 实际转化为s.set_score(60)
