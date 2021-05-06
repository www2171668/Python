
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


cifar = keras.datasets.cifar10


# In[3]:


(train_image, train_label), (test_image, test_label) = cifar.load_data()


# In[4]:


train_image.shape, test_image.shape


# In[5]:


train_label


# 神经网络: 最好做归一化的处理

# In[6]:


train_image = train_image/255


# In[7]:


test_image = test_image/255


# In[8]:


model = keras.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (1, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))          
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128))
model.add(layers.BatchNormalization())  
model.add(layers.Dropout(0.5))         
model.add(layers.Dense(10, activation='softmax'))


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)


# In[11]:


history = model.fit(train_image, train_label, epochs=30, batch_size=128)


# In[12]:


model.evaluate(test_image, test_label)


# In[24]:


model.evaluate(test_image, test_label)

