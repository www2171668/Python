
# coding: utf-8

# In[1]:


import keras
from keras import layers
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


import keras.datasets.mnist as mnist


# In[3]:


(train_image, train_label), (test_image, test_label) = mnist.load_data()


# In[4]:


train_image.shape


# In[9]:


plt.imshow(train_image[1000])


# In[7]:


train_label.shape


# In[10]:


train_label[1000]


# In[12]:


test_image.shape, test_label.shape


# In[13]:


train_label


# In[24]:


model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[25]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[27]:


model.fit(train_image, train_label, epochs=50, batch_size=512)


# In[28]:


model.evaluate(test_image, test_label)


# In[29]:


model.evaluate(train_image, train_label)


# In[30]:


import numpy as np


# In[32]:


np.argmax(model.predict(test_image[:10]), axis=1)


# In[33]:


test_label[:10]


# 模型的优化

# In[34]:


model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[35]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[37]:


model.fit(train_image, train_label, epochs=50, batch_size=512, validation_data=(test_image, test_label))


# 模型的再优化

# In[52]:


model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


# In[53]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[54]:


model.fit(train_image, train_label, epochs=200, batch_size=512, validation_data=(test_image, test_label))

