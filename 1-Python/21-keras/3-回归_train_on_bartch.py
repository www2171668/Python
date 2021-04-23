""""""

import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# %% 构造数据 + 数据分割
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # \randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

plt.scatter(X, Y)   # \plot data
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# %% 构建网络   Sequential() 初始化网络       Sequential().add 增加层       Dense() 全连接层
model = Sequential()
model.add(Dense(units=1, input_dim=1))  # * Dense(units,input_dim)  神经元数量（输出维度），输入维度

#%% 编译   Sequential().compile(loss ，optimizer ，metrics ) 定义优化器（损失函数和优化方法）        model.summary() 显示模型层和参数信息
model.compile(loss='mse', optimizer='sgd')  # \choose loss function and optimizing method

model.summary()

# %% 训练    model.train_on_batch(X,Y) 批量训练
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# %% 测试    model.evaluate(X,Y,batch) 评估损失
loss = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', loss)

# %% 得到网络权重    net.get_weights()
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# %% 预测  model.predict(X)
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
