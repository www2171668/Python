import numpy as np
import keras
from keras.models import load_model

import keras.datasets.mnist as mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()
test_image = np.expand_dims(test_image, axis=-1)

# %% 编码器也会被读取
my_model = load_model('my_model.h5')
my_model.evaluate(test_image, test_label)
my_model.predict(test_image[0:2])

# %% 先创建编码器，再加载模型参数。顺序没有要求
from keras.models import model_from_json

with open('my_json_model.json') as f:
    my_json_model = f.read()
model = model_from_json(my_json_model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('my_model_weights.h5')
model.evaluate(test_image, test_label)

# %%
from keras import layers

model_new = keras.Sequential()
model_new.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_1'))
model_new.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv_2'))
model_new.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_new.add(layers.Flatten())
model_new.add(layers.Dense(256, activation='relu', name='dense_1_'))
model_new.add(layers.Dropout(0.5))
model_new.add(layers.Dense(10, activation='softmax', name='dense_2_'))

model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_new.load_weights('my_model_weights.h5', by_name=True)
model_new.evaluate(test_image, test_label)
