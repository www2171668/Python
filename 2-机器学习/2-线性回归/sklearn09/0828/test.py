# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/28
"""
# TODO: 这个代码不能运行
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

x = [[1, 2, 3]]
y = [2.0]
poly = PolynomialFeatures()
# 这里的y在实际中代码执行中是不会用到的。此时你传不传y都没有关系
# 所以这里把函数API设置为x、y这种形式，只为为了API统一。
x = poly.fit_transform(x, y)
print(x)

lr = LinearRegression()
# 这里进行模型训练，必须传入x和y
lr.fit(x, y)
print(lr.predict(x))

pipline = Pipeline(steps=[
    ('s1', PolynomialFeatures(degree=3)),
    ('s2', LinearRegression())
])
# 管道流调用fit的时候，其实相当于每一步调用fit_transform(所有的特征工程)或者fit API(最后一步模型训练)
pipline.fit(x, y)
print(pipline.predict(x))
