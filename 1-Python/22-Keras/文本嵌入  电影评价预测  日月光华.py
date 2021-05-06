
# coding: utf-8

# In[1]:


import keras
from keras import layers


# In[2]:


data = keras.datasets.imdb


# In[3]:


max_word = 10000


# In[11]:


(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)


# In[12]:


x_train.shape, y_train.shape


# In[13]:


maxlen = 200


# In[14]:


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[16]:


x_train.shape


# In[17]:


model = keras.Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))


# In[18]:


model.output_shape


# In[19]:


model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[20]:


model.summary()

