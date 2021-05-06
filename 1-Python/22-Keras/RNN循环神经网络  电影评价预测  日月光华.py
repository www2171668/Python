
# coding: utf-8

# In[1]:


import keras
from keras import layers


# In[2]:


data = keras.datasets.imdb


# In[3]:


max_word = 10000


# In[4]:


(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)


# In[5]:


x_train.shape, y_train.shape


# In[6]:


maxlen = 200


# In[7]:


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[10]:


x_train.shape, x_test.shape


# In[11]:


model = keras.Sequential()
model.add(layers.Embedding(10000, 20, input_length=maxlen))


# In[12]:


model.output_shape


# In[13]:


model.input_shape


# In[14]:


model.add(layers.LSTM(128))


# In[15]:


model.output_shape


# In[16]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[17]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)


# In[18]:


history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=256, 
                    validation_data=(x_test, y_test))

