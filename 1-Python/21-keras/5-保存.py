""""""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#%% data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

#%% 网络
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

#%% 训练
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)

#%% 保存  model.save('path_file') HDF5 file(pip3 install h5py)
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')
del model  # * deletes the existing model

#%% 读取  load_model('path_file')
model = load_model('mpathy_model.h5')
print('test after load: ', model.predict(X_test[0:2]))

#%% 保存和读取权重     model.save_weights() / model.load_weights()
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

#%% 保存和读取结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
