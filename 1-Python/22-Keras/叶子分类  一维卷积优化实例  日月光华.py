
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv('./dataset/leaf/train.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


len(data)


# In[6]:


len(data.species.unique())


# In[7]:


labels = pd.factorize(data.species)[0]


# In[8]:


x = data[data.columns[2:]]


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


train_x, test_x, train_y, test_y = train_test_split(x, labels)


# In[11]:


train_x.shape, test_x.shape


# In[12]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)


# In[13]:


train_x = (train_x - mean)/std
test_x = (test_x - mean)/std


# In[14]:


train_x.shape


# 把每一条数据看成一个序列

# 一维卷积或者LSTM： samples, step, feature

# In[15]:


train_x = np.expand_dims(train_x, -1)


# In[16]:


train_x.shape


# In[17]:


test_x = np.expand_dims(test_x, -1)


# In[18]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 7, input_shape=(train_x.shape[1:]), activation='relu', padding='valid'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 7, activation='relu', padding='valid'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(99, activation="softmax"))


# In[19]:


model.summary()


# In[20]:


model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[21]:


history = model.fit(train_x, train_y,
                   epochs=600,
                   batch_size=128,
                   validation_data=(test_x, test_y))


# In[23]:


plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()

