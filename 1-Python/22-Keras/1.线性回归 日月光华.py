import keras

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 30)
y = 3 * x + 7 + np.random.randn(30) * 6

plt.scatter(x, y)
model = keras.Sequential()  # 顺序模型

from keras import layers

model.add(layers.Dense(1, input_dim=1))

model.summary()
# 编译模型
model.compile(optimizer='adam', loss='mse')
# 训练模型
model.fit(x, y, epochs=3000)
model.predict(x)

plt.scatter(x, y, c='r')
plt.plot(x, model.predict(x))

model.predict([150])
