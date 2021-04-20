# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/25
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 加载数据
path = '../datas/household_power_consumption_200.txt'  ## 200行数据
path = '../datas/household_power_consumption_1000.txt'  ## 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
names2 = df.columns
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')  # 只要有数据为空，就进行删除操作

# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]
X = X.astype(np.float)
Y = Y.astype(np.float)

# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 1. 构建一个管道流对象，定义数据处理的顺序
"""
Ridge参数说明: 
  alpha=1.0: ppt上的λ，正则化系数, 
  fit_intercept=True: 模型是否训练截距项，默认为训练(True), 
  normalize=False：在做模型训练之前，是否做归一化操作，一般不改动,
  max_iter=None：模型求解的迭代最大次数，默认表示不限制, 
  tol=1e-3: 模型收敛条件，当损失函数的变化值小于该值的时候，介绍迭代更新, 
  solver="auto"：给定求解方式,
RidgeCV:
  alphas: 给定alpha的取值范围
  cv: 给定做几折交叉验证
"""
model = Pipeline(steps=[
    ('Poly', PolynomialFeatures()),  # 给定第一步操作，名称为Poly
    ('Linear', RidgeCV(alphas=[0.1, 0.2, 0.3], cv=5))  # 给定第二步操作，名称为Linear
])

# 1.2 Pipeline对象设置参数
# Poly__degree: Poly是定义Pipeline对象的时候给定的步骤名称，degree是对应步骤对象中的属性名称, 中间是两个连续的下划线
model.set_params(Poly__degree=2)
model.set_params(Linear__normalize=True)

# 2. 模型训练（先调用第一步进行数据处理，然后再调用第二步做模型训练）
# 假设我们的步骤是n步，那么前n-1步做的操作是: fit + transform， 最后一步做的操作是fit
model.fit(X_train, Y_train)
"""
model.fit等价于linear.fit(poly.fit_transform(x_train,y_train),y_train)
"""

print("多项式模型:{}".format(model.get_params()['Poly']))
print("线性回归模型:{}".format(model.get_params()['Linear']))
# 3. 预测值产生(先调用第一步的transform对数据转换，再调用第二步的predict对数据预测)
# 假设我们的步骤是n步，那么前n-1步做的操作是: transform， 最后一步做的操作是fit
y_hat = model.predict(X_test)

# 7.1 查看训练好的模型参数
linear_algo = model.get_params()['Linear']
print("线性回归的各个特征属性对应的权重参数θ:{}".format(linear_algo.coef_))
print("线性回归的截距项的值:{}".format(linear_algo.intercept_))

# 八、模型效果评估
print("在训练集上的模型效果(回归算法中为R2):{}".format(model.score(X_train, Y_train)))
print("在测试集上的模型效果(回归算法中为R2):{}".format(model.score(X_test, Y_test)))
print("在测试集上的MSE的值:{}".format(mean_squared_error(y_true=Y_test, y_pred=y_hat)))
print("在测试集上的RMSE的值:{}".format(np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_hat))))

# 画图看一下效果
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
