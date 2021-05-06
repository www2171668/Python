
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')


# In[4]:


data.info()


# In[5]:


data[data['pm2.5'].isna()]


# In[6]:


data = data.iloc[24:].copy()


# In[7]:


data.fillna(method='ffill', inplace=True)


# In[8]:


data.info()


# In[9]:


data.drop('No', axis=1, inplace=True)


# In[10]:


import datetime


# In[11]:


data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'],
                                       month=x['month'],
                                       day=x['day'], 
                                       hour=x['hour']), 
                          axis=1)


# In[12]:


data.set_index('time', inplace=True)


# In[13]:


data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)


# In[14]:


data.head()


# In[15]:


data.columns = ['pm2.5', 'dew', 'temp', 'press', 'cbwd', 'iws', 'snow', 'rain']


# In[16]:


data.cbwd.unique()


# In[17]:


data = data.join(pd.get_dummies(data.cbwd))


# In[18]:


del data['cbwd']


# In[19]:


data.info()


# In[20]:


data['pm2.5'][-1000:].plot()


# In[21]:


data['temp'][-1000:].plot()


# In[22]:


data.head(3)


# In[23]:


sequence_length = 5*24
delay = 24


# In[24]:


data_ = []
for i in range(len(data) - sequence_length - delay):
    data_.append(data.iloc[i: i + sequence_length + delay])


# In[25]:


data_ = np.array([df.values for df in data_])


# In[26]:


data_.shape


# In[27]:


np.random.shuffle(data_)


# In[28]:


x = data_[:, :-delay, :]
y = data_[:, -1, 0]


# In[29]:


split_boundary = int(data_.shape[0] * 0.8)


# In[30]:


train_x = x[: split_boundary]
test_x = x[split_boundary:]

train_y = y[: split_boundary]
test_y = y[split_boundary:]


# In[31]:


train_x.shape, test_x.shape, train_y.shape, test_y.shape


# In[32]:


mean = train_x.mean(axis=0)
std = train_x.std(axis=0)


# In[33]:


mean.shape


# In[34]:


train_x = (train_x - mean)/std


# In[35]:


test_x = (test_x - mean)/std


# In[36]:


batch_size = 128


# In[37]:


model = keras.Sequential()
model.add(layers.Flatten(input_shape=(train_x.shape[1:])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))


# In[38]:


model.compile(optimizer=keras.optimizers.Adam(), loss='mae')
history = model.fit(train_x, train_y,
                    batch_size = 128,
                    epochs=50,
                    validation_data=(test_x, test_y)
                    )


# In[39]:


plt.plot(history.epoch, history.history.get('loss'), 'y', label='Training loss')
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.legend()


# In[46]:


model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(train_x.shape[1:])))
model.add(layers.Dense(1))


# In[47]:


model.compile(optimizer=keras.optimizers.Adam(), loss='mae')


# In[48]:


history = model.fit(train_x, train_y,
                    batch_size = 128,
                    epochs=200,
                    validation_data=(test_x, test_y))


# In[49]:


plt.plot(history.epoch, history.history.get('loss'), 'y', label='Training loss')
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.legend()


# In[68]:


model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(1))


# In[69]:


model.compile(optimizer=keras.optimizers.Adam(), loss='mae')


# In[70]:


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)


# In[71]:


history = model.fit(train_x, train_y,
                    batch_size = 128,
                    epochs=200,
                    validation_data=(test_x, test_y),
                    callbacks=[learning_rate_reduction])

