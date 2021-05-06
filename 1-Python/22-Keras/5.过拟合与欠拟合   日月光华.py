
# coding: utf-8

# In[1]:


import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('./dataset/credit-a.csv', header=None)


# In[3]:


data.iloc[:, -1].unique()


# In[4]:


x = data.iloc[:, :-1].values


# In[5]:


y = data.iloc[: , -1].replace(-1, 0).values.reshape(-1, 1)


# In[6]:


y.shape, x.shape


# In[7]:


model = keras.Sequential()
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[8]:


model.summary()


# In[9]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[10]:


history = model.fit(x, y, epochs=1000)


# In[11]:


history.history.keys()


# In[12]:


plt.plot(history.epoch, history.history.get('loss'), c='r')
plt.plot(history.epoch, history.history.get('acc'), c='b')


# 评价标准： 对未见过数据的预测

# In[14]:


x_train = x[:int(len(x)*0.75)]
x_test = x[int(len(x)*0.75):]
y_train = y[:int(len(x)*0.75)]
y_test = y[int(len(x)*0.75):]


# In[15]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[28]:


model = keras.Sequential()


# In[29]:


model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[30]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[31]:


history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))


# In[32]:


plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()


# In[22]:


model.evaluate(x_train, y_train)


# In[23]:


model.evaluate(x_test, y_test)


# 过拟合：在训练数据正确率非常高， 在测试数据上比较低

# In[33]:


model = keras.Sequential()
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[34]:


model.summary()


# In[35]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[36]:


history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))


# In[37]:


model.evaluate(x_train, y_train)


# In[38]:


model.evaluate(x_test, y_test)


# In[39]:


plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()


# 正则化 

# l1     
# loss =  s*abs(w1 + w2 + ..) + mse
# 
# l2   
# loss =  s*(w1**2 + w2**2 + ..) + mse

# In[40]:


from keras import regularizers


# In[41]:


model = keras.Sequential()
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), input_dim=15, activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[42]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[43]:


history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))


# In[44]:


model.evaluate(x_train, y_train)


# In[45]:


model.evaluate(x_test, y_test)


# In[46]:


model = keras.Sequential()
model.add(layers.Dense(4, input_dim=15, activation='relu'))
model.add(layers.Dense(4,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[47]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[48]:


history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))


# In[49]:


model.evaluate(x_train, y_train)


# In[50]:


model.evaluate(x_test, y_test)


# In[19]:


model = keras.Sequential()
model.add(layers.Dense(4, input_dim=15, activation='relu'))
model.add(layers.Dense(1,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[20]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[21]:


history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))

