
# coding: utf-8

# In[1]:


import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


import keras.datasets.mnist as mnist


# In[5]:


(train_image, train_label), (test_image, test_label) = mnist.load_data()


# In[7]:


train_image.shape, test_image.shape


# 图像的数据的shape

# hight  witch   chanel
# 
# 彩色图像：  (h, w, 1)
# 
# conv2d    :   图片输入的形状： batch, height, width, channels 
# 
# dense    :               batch, data

# In[9]:


a = np.array([1, 2, 3])


# In[13]:


a.ndim


# In[14]:


np.expand_dims(a, axis=-1).ndim


# In[15]:


train_image = np.expand_dims(train_image, axis=-1)


# In[16]:


train_image.shape


# In[17]:


test_image = np.expand_dims(test_image, axis=-1)


# In[18]:


test_image.shape


# In[19]:


model = keras.Sequential()


# In[20]:


model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))


# In[21]:


model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[22]:


model.add(layers.MaxPooling2D())


# In[23]:


model.summary()


# In[24]:


model.add(layers.Flatten())


# In[25]:


model.summary()


# In[26]:


model.add(layers.Dense(256, activation='relu'))


# In[27]:


model.add(layers.Dropout(0.5))


# In[28]:


model.add(layers.Dense(10, activation='softmax'))


# In[29]:


model.summary()


# In[31]:


train_label


# In[32]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[ ]:


model.fit(train_image, train_label, epochs=10, batch_size=512)

