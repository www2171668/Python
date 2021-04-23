""""""

# %%  def 函数名([参数]):

def Pname():
    print('大家好我是小明同学！')

Pname()  # \ 函数调用方法一
pri = Pname  # \ 函数调用方法二
pri()

# %%  定长形参，int str
# 1、必备参数
def getInfo(name, address):
    print('大家好我叫%s，我来自%s' % (name, address))

def getInfoplus(name, address='香港'):
    print('大家好我叫%s，我来自%s' % (name, address))

getInfo('刘德华', '香港')
getInfo(name='刘德华', address='香港')
getInfoplus('刘德华')

# %% 不定长形参一  ★    *args（arguments）：tuple类型不定长参数     **kwargs（keyword arguments）：dict类型不定长参数
def getInfo(first, *args, **kwargs):
    print('argument: ', first)
    for v in args:
        print('Optional argument (args): ', v)
    for k, v in kwargs.items():
        print('Optional argument %s (kwargs): %s' % (k, v))

getInfo(0, 1, 2, 3, k1=4, k2=5)  # 调用函数：在调用时参数被自动打包

args = [1, 2, 3]
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
getInfo(0, args, kwargs)  # * 使用解包调用函数的代码。实际上，使用打包好的参数无需类()中使用*和**来表示，即可以简化为普通形参

# %%  值传递：str tuple 的传递       引用传递：list dict 的传递，在函数中修改传递值，原值改变  ※
# 1、值传递：不可变对象的传递
def fun(args):
    args = 'hello'  # \ 重新赋值
    print(args)

str1 = 'baby'  # 不可变数据类型
fun(str1)
print(str1)  # \ 没有被改变

# 2、引用传递：可变对象的传递
def fun(args):
    args[0] = 'hello'  # \ 重新赋值
    print(args)

list01 = ['baby', 'come on']  # 可变数据类型
fun(list01)
print(list01)  # \ 传递的是对象本身，函数里面被修改了值，原对象也会跟着修改

# %% return *   返回*，执行到return时，后面的代码都不会执行

# return返回多个返回值 ★
def get_sum(x, y):
    return x, y

num = get_sum(1, 2)  # 1、单变量接收多返回值,返回值会保存在元组中
num1, num2 = get_sum(1, 2)  # 2、多变量接收多返回值
print(num)
print(num1, num2)

# %% argparse：用于解析命令行参数和选项的标准模块
"""
    使用argparse：
        ①、import argparse
        ②、parser = argparse.ArgumentParser()   创建一个解析对象
        ③、parser.add_argument()   用来指定程序需要接受的命令参数，每一个add_argument方法对应一个参数
        ④、parser.parse_args()   对argparse进行解析（调用），通过 .name 来调用add_argument()中的命令参数

    .ArgumentParser(description):
        description：描述程序

    .add_argument(name,default,type,choices,help)：
        name：①、通过 - 来指定的短参数，如 -algorithm
              ②、通过 -- 来指定的长参数，如 --algorithm
        default：参数默认值
        type：参数类型，默认为str
        choices = []：限定取值范围
        help：解释说明，无实际功能
"""

import argparse  # ①

parser = argparse.ArgumentParser(description="calculate X to the power of Y")  # ②
parser.add_argument("-square", default=1, type=int, help="display a square of a given number")  # ③
parser.add_argument("--verbosity", type=int, default=1, help="increase output verbosity")
args = parser.parse_args()  # ④

answer = args.square ** 2  # 通过args.square对参数进行调用
if args.verbosity == 2:
    print("the square of {} equals {}".format(args.square, answer))
else:
    print(answer)
