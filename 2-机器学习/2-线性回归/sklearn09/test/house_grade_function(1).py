import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
"""
    function: thegema(theta1 * x1 + theta2 *x2)*x1
"""


class LinearTrain:
    """
    线性回归拟合y = theta0 + theta1 * x1 + theta2 * x2
    alpha: 下降率
    loop_max: 最大循环次数
    f_change: 收敛范围
    """

    def __init__(self, alpha=0.0005, loop_max=100, f_change=1e-10):
        self.alpha = alpha
        self.loop_max = loop_max
        self.f_change = f_change  # 收敛程度
        self.theta = np.random.randn(3)
        # print(id(self.theta))
        """
        训练theta值
        x: 训练样本x
        y: 训练样本y
        """

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        iter_num = 1
        error = np.zeros(3)
        while iter_num < self.loop_max:
            iter_num += 1
            # self.__random_h0(x, y) #随机梯度下降
            # self.__minibatch_h0(x, y, 5) #小批量梯度下降
            self.__batch_h0(x, y)  # 批量梯度下降
            print("循环外的theta:", id(self.theta), self.theta)
            if np.linalg.norm(self.theta - error) < self.f_change:  # 判断是否收敛到规定值
                print("值:", np.linalg.norm(self.theta - error))
                break
            else:
                error = self.theta
                # print("error:", id(error), error, id(self.theta), self.theta)
        print(self.theta)
        print("迭代次数：", iter_num)

    """
    转换函数
    x: 训练样本集
    """

    def transfrom(self, x):
        x = np.mat(x)
        y = np.mat(self.theta).reshape(-1, 1)
        return x * y

    """
    预测函数
    x: 预测数据集
    """

    def predict(self, x):
        x = np.mat(x)
        y = np.mat(self.theta).reshape(-1, 1)
        return x * y

    def __batch_h0(self, x, y):
        print("")
        print("更新之前的theta:", self.theta)
        value = np.zeros(3)
        for i in np.arange(160):
            a = self.theta
            b = x[i]
            c = y[i]
            value += (np.dot(a, b) - c) * b
            # value += (np.dot(self.theta, x[i]) - y[i]) * x[i]
        # self.theta = self.theta - self.alpha * value
        self.theta -= self.alpha * value
        print(self.theta)
        # print(id(self.theta))
        # print(self.theta)

    def __random_h0(self, x, y):
        for i in np.arange(160):
            self.theta = self.theta - self.alpha * (np.dot(self.theta, x[i]) - y[i]) * x[i]

    def __minibatch_h0(self, x, y, minibatch_size):
        for i in np.arange(1, x.shape[0], minibatch_size):
            sum_m = np.zeros(3)
            for j in np.arange(i - 1, i + minibatch_size - 1, 1):
                dif = (np.dot(self.theta, x[j]) - y[j]) * x[j]
                sum_m += dif
            self.theta = self.theta - self.alpha * (1.0 / minibatch_size) * sum_m


path1 = './datas/household_power_consumption_200.txt'
df = pd.read_csv(path1, sep=';')
X = df[['Global_active_power', 'Global_reactive_power']]
m = X.shape[0]
x0 = np.full(m, 1.0)
X.insert(0, 'theta', x0)
Y = df['Global_intensity']
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8, random_state=28)

# 对数据集标准化和不标准化的区别
# ss = StandardScaler()
# train_x = ss.fit_transform(train_x)
# test_x = ss.transform(test_x)

lr = LinearTrain(0.00009)
lr.fit(train_x, train_y)
y_hat = lr.transfrom(test_x)

print(y_hat[:2])
print(test_y[:2])
t = np.arange(len(test_x))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, test_y, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
