
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


data.species.unique()


# In[6]:


len(data.species.unique())


# In[7]:


labels = pd.factorize(data.pop('species'))


# In[8]:


y = labels[0]


# In[9]:


_ = data.pop('id')


# In[10]:


x = data


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


train_x, test_x, train_y, test_y = train_test_split(x, y)


# In[13]:


train_x.shape, test_x.shape


# In[14]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)


# In[15]:


train_x_norm = (train_x - mean)/std


# In[16]:


test_x_norm = (test_x - mean)/std


# In[17]:


train_x_norm = np.expand_dims(train_x_norm, -1)


# In[18]:


train_x_norm.shape


# In[19]:


test_x_norm = np.expand_dims(test_x_norm, -1)


# In[20]:


test_x_norm.shape


# In[110]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(99, activation='softmax'))


# In[111]:


model.summary()


# In[112]:


train_x_norm.shape


# In[113]:


model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[114]:


history = model.fit(train_x_norm, train_y, epochs=1000, batch_size=128, validation_data=(test_x_norm, test_y))


# In[115]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[126]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))


# In[127]:


model.summary()


# In[128]:


model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[129]:


history = model.fit(train_x_norm, train_y, epochs=1000, batch_size=128, validation_data=(test_x_norm, test_y))


# In[130]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[116]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(256, 7, activation='relu', padding='same'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))


# In[117]:


model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[120]:


model.summary()


# In[118]:


history = model.fit(train_x_norm, train_y, epochs=1000, batch_size=128, validation_data=(test_x_norm, test_y))


# In[119]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[188]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256, 7, activation='relu', padding='same'))
model.add(layers.Conv1D(256, 7, activation='relu', padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))


# In[189]:


model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[190]:


history = model.fit(train_x_norm, train_y, 
                    epochs=1000,  
                    validation_data=(test_x_norm, test_y))


# In[191]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[192]:


plt.plot(history.epoch, history.history.get('loss'), 'y', label='Training loss')
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.legend()


# In[194]:


model = keras.Sequential()
model.add(layers.Conv1D(32, 11, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(32, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(64, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[195]:


model.summary()


# In[196]:


history = model.fit(train_x_norm, train_y, 
                    epochs=1000,  
                    validation_data=(test_x_norm, test_y))


# In[197]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[198]:


model = keras.Sequential()
model.add(layers.Conv1D(64, 11, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(64, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(512, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(512, 11, activation='relu', padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[199]:


history = model.fit(train_x_norm, train_y, 
                    epochs=1000,  
                    validation_data=(test_x_norm, test_y))


# In[200]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[202]:


model = keras.Sequential()
model.add(layers.Conv1D(64, 11, activation='relu', padding='same', input_shape=train_x_norm.shape[1:]))
model.add(layers.Conv1D(64, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(128, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(256, 11, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(512, 11, activation='relu', padding='same'))
model.add(layers.Conv1D(512, 11, activation='relu', padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(99, activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[203]:


history = model.fit(train_x_norm, train_y, 
                    epochs=1000,  
                    validation_data=(test_x_norm, test_y))


# In[204]:


plt.plot(history.epoch, history.history.get('acc'), 'y', label='Training acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b', label='Test acc')
plt.legend()


# In[206]:


max(history.history.get('acc'))


# In[207]:


max(history.history.get('val_acc'))

