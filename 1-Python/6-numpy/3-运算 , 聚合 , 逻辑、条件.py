""""""
import numpy as np

# %% 某一列的操作
num = np.arange(1, 5).reshape(2, 2)
num[:, 0] = num[:, 0] * 100  # 注意赋值
print(num)

# %% 数组与标量 / 数组与数组的加法
num = np.arange(1, 5).reshape(2, 2)
print(num + 2)

arr1 = np.array([[1, 2], [2, 3]])
arr2 = np.array([[5, 6], [6, 7]])
print(arr1 + arr2)

# %% * 点乘        np.dot(*,*) 矩阵叉乘，矩阵乘法
arr1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
arr2 = np.array([[2, 2, 3], [2, 2, 3], [2, 2, 3]])

print(arr1 * arr2)  # 点乘
print(np.dot(arr1, arr2))

# %% 计算方法   np.abs() / np.fabs()        np.power(arr, m) /  np.square()     np.sqrt()   np.exp()    np.log()    np.std() 标准差
arr = np.arange(12).reshape(4, 3)
print(arr)

print(np.abs(arr))
print(np.power(arr, 2))  # 2次指数运算
print(np.sqrt(arr))
print(np.square(arr))
print(np.exp(arr))  # 计算各个元素的指数e的x次方。自然底数e： e ≈ 2.71828
print(np.log1p(arr))  # 计算底数为10的log；np.log1p(),底数为e

a = np.array([[5, 2, 3, 4], [7, 3, 9, 1]])
print(np.std(a))  # \ np.mean((a - a.mean()) ** 2)
# %% 数字类型
"""
    —— np.sign() 计算各个元素的正负号: 1  正数，0：零，-1：负数
    —— np.ceil() / np.floor() 向上取整,向下取整
    —— np.around(a, decimals) 四舍五入  a: 输入的数组    decimals: 要舍入的小数位，默认值位0，负数讲四舍五入到小数点左侧位置
    —— np.isnan()返回一个区分“NaN(不是一个数字)”的布尔类型数组。NaN为True
    —— np.isinf()返回一个区分有穷数的布尔类型数组。无穷数为True
"""
num = np.random.randn(1, 9).reshape(3, 3)
print(np.sign(num))

arr = np.array([-1.7, 1.5, -0.1, 0.6, 10])
print(np.ceil(arr))  # \ [-1.  2. -0.  1. 10.] 输出的还是二进制

arr = np.array([-1.78, 1.5, -0.1, 12.6, 10])
print(np.around(arr))  # \到个位 [-1.  2. -0.  1. 10.] 输出的还是二进制
print(np.around(arr, 1))  # \ 到小数点后一位
print(np.around(arr, -1))  # \ 到十位

arr = np.random.random(6).reshape(2, 3)
print(np.isnan(arr))
print(np.isfinite(num))

# %% 聚合函数 np.min(﹡) / np.max(﹡) / np.mean(﹡)          ﹡.min() / ﹡.max() / ﹡.mean() 不建议
# axis = 0 按列遍历，返回每一列中的最小 / 大值组成的数组 ★
# axis = 1 按行遍历，返回每一行中的最小 / 大值组成的数组
a = np.array([[5, 2, 3, 4], [7, 3, 9, 1]])
print(a)
print('min:', a.min())
print('min:', a.min(axis=0))
print('min:', a.min(axis=1))
print('min:', np.min(a, axis=0))

print('min:', a[:, 0].min())  # 通常用切片方法定位好，再进行函数处理。其他的函数也类似

# %% 三角函数 *np.pi / 180  标准三角函数转换方式，使角度转化为弧度。np.pi就是π
a = np.array([0, 30, 45, 60, 90])
print(np.sin(a * np.pi / 180))

print(np.arcsin(np.sin(a * np.pi / 180)))

print(np.degrees(np.arctan(np.tan(a * np.pi / 180))))

# %% 比较运算，返回一个布尔型数组 np.greater(﹡, ﹡) / np.less(﹡, ﹡) / np.equal(﹡, ﹡) 大于
num1 = np.random.randint(1, 7, size=(2, 3))
num2 = np.random.randint(1, 7, size=(2, 3))
print(num1)
print(num2)
print(np.greater(num1, num2))

# %% 条件函数 np.where([condition])  一元表达式，输出为True的索引值      不建议使用，效果不好
a = np.array([1, 2], [9, 8])
b = np.where(a < 5)  # \ 返回索引
print(b, np.shape(b))
print(b[0][0])

# %% 条件函数 np.where([condition], [x], [y])  三元表达式   x if condition else y 的矢量化版本
a = np.where([True, False], [1, 2], [9, 8])
print(a)

a = np.where([[True, False], [False, False]], [[1, 2], [3, 4]], [[9, 8], [7, 6]])
print(a)

# %% np.unique() 元素去重
arr = np.array(['图书', '数码', '小吃', '数码', '男装', '小吃', '美食', '数码', '女装'])
print(np.unique(arr))
# %% |  &  ！  或且非
