""""""

import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# %% data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# %% 构造网络    Sequential([],[],...)
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),  # * Activation(‘’)
    Dense(10),
    Activation('softmax'),
])

# %% 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])  # add metrics to get more results you want to see

# %% 训练 model.fit(X,Y,epochs,batch)
print('Training ------------')
model.fit(X_train, y_train, epochs=2, batch_size=32)

# %% 测试
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)  # 原本只输出loss，现在增加了accuracy
print('test loss: ', loss)
print('test accuracy: ', accuracy)
