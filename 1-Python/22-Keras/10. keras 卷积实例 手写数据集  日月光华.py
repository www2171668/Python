
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras import layers


# In[2]:


import keras.datasets.mnist as mnist


# In[3]:


(train_image, train_label), (test_image, test_label) = mnist.load_data()


# In[4]:


train_image.shape


# In[5]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(train_image[4])


# In[6]:


train_label[4]


# In[7]:


import numpy as np


# In[8]:


train_image = np.expand_dims(train_image, axis=-1)


# In[9]:


train_image.shape


# In[10]:


train_label.shape


# In[11]:


test_image.shape


# In[12]:


test_image = np.expand_dims(test_image, axis=-1)


# In[13]:


#初始化模型


# In[14]:


model = keras.Sequential()


# In[15]:


#添加层，构建网络


# In[16]:


model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_1'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv_2'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', name='dense_1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax', name='dense_2'))


# In[17]:


model.summary()


# In[18]:


#编译模型 


# In[19]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# tf.nn.sparse_softmax_cross_entropy_with_logits  
# tf.nn.softmax_cross_entropy_with_logits_v2   # one-hot

# In[20]:


#训练模型


# In[21]:


model.fit(train_image, train_label, epochs=3, batch_size=512)


# In[22]:


model.evaluate(test_image, test_label)


# In[26]:


model.save_weights('my_model_weights.h5')


# In[25]:


import numpy as np
np.argmax(model.predict(test_image[:10]), axis=1)


# In[26]:


test_label[:10]


# 模型的优化

# In[23]:


model = keras.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


# In[24]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[25]:


model.fit(train_image, train_label, epochs=10, batch_size=512)


# In[26]:


model.evaluate(test_image, test_label)


# In[33]:


model.save('my_model.h5')


# In[35]:


my_model_json = model.to_json()


# In[36]:


my_model_json


# In[37]:


with open('my_json_model.json', 'w') as f:
    f.write(my_model_json)

