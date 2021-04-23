""""""
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# %% 数据
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

encoding_dim = 2    # plot in a 2D figure

# %% input placeholder and encoder layers
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# %% decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded) # [-1,1]

# %% autoencoder model     Model(input,output) Input和最后一个网络层构成了整个网络
autoencoder = Model(input=input_img, output=decoded)

# %% encoder model (for plotting)
encoder = Model(input=input_img, output=encoder_output)

# %% 编译
autoencoder.compile(optimizer='adam', loss='mse')

# %% 训练
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128)

# %% plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()
