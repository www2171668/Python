""""""


# %% 继承    继承时，左边出现o符号
# 父类名.__init__(self,父类变量) ：如果父类有重写的构造函数，子类一定要在其构造函数中进行加载；如果没有，则不加载
# 多态：重写父类中的方法

class Animal:
    def __init__(self, name, food):
        self.name = name
        self.food = food

    def eat(self):
        print('%s爱吃%s' % (self.name, self.food))


class Dog(Animal):  # 多个继承用,分割
    def __init__(self, name, food, drink):
        # * 加载父类构造方法，获得父类的实例属性
        # super().__init__(name,food)         #方法一
        # super(Dog, self).__init__(name,food)         #方法二
        Animal.__init__(self, name, food)  # 方法三

        self.drink = drink  # 子类属性

    def drinks(self):
        print('%s爱喝%s' % (self.name, self.drink))

    # def eat(self):      # 多态，即重写
    #     print('%s不爱吃%s' % (self.name, self.food))


dog1 = Dog('金毛', '骨头', '可乐')
dog1.eat()
dog1.drinks()


# %%嵌套类
def X(Q):
    def Y(y):
        t = Q[1]
        return t + y

    return Y  # * 返回的是Y(y)这个方法对象


Q = [1, 2, 3]
c = X(Q)  # 接收Y(y)方法对象，此时c等价于Y的实例化方法
d = c(10)  # 使用Y(y)方法
print(d)  # \ 12
