""""""

import numpy as np

np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# %% data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# %% 网络
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

# %% 训练
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)

# %% 一、保存模型  model.save('path_file') HDF5 file(pip3 install h5py)
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')
del model  # * deletes the existing model

# %% 读取模型  load_model('path_file')
model = load_model('mpathy_model.h5')
print('test after load: ', model.predict(X_test[0:2]))

# %% 二、保存权重     model.save_weights('path_file')
model.save_weights('my_model_weights.h5')

# %% 读取权重    model.load_weights('path_file')
model.load_weights('my_model_weights.h5')

# %% 三、保存模型结构
json_string = model.to_json()

# %% 读取模型结构
from keras.models import model_from_json

with open('my_json_model.json') as f:
    my_json_model = f.read()
model = model_from_json(my_json_model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('my_model_weights.h5')

model.evaluate(X_test, Y_test)

