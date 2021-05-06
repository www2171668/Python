
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
import shutil


# In[2]:


base_dir = './dataset/cat_dog'
train_dir = base_dir + '/train'
train_dog_dir = train_dir + '/dog'
train_cat_dir = train_dir + '/cat'
test_dir = base_dir + '/test'
test_dog_dir = test_dir + '/dog'
test_cat_dir = test_dir + '/cat'
dc_dir = './dataset/dc/train' 


# In[3]:


if not os.path.exists(base_dir):

    os.mkdir(base_dir)

    os.mkdir(train_dir)
    os.mkdir(train_dog_dir)
    os.mkdir(train_cat_dir)
    os.mkdir(test_dir)
    os.mkdir(test_dog_dir)
    os.mkdir(test_cat_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 
    for fname in fnames:
        src = os.path.join(dc_dir, fname)
        dst = os.path.join(train_cat_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)] 
    for fname in fnames:
        src = os.path.join(dc_dir, fname)
        dst = os.path.join(test_cat_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)] 
    for fname in fnames:
        src = os.path.join(dc_dir, fname)
        dst = os.path.join(train_dog_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)] 
    for fname in fnames:
        src = os.path.join(dc_dir, fname)
        dst = os.path.join(test_dog_dir, fname)
        shutil.copyfile(src, dst)


# In[4]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)                        
test_datagen = ImageDataGenerator(rescale=1./255)


# In[5]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(200, 200),
    batch_size=20,
    class_mode='binary'
)


# keras内置经典网络实现

# In[6]:


covn_base = keras.applications.VGG16(weights='imagenet', include_top=False)


# In[7]:


covn_base.summary()


# In[8]:


model = keras.Sequential()
model.add(covn_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[9]:


model.summary()


# In[10]:


covn_base.trainable = False


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])


# In[13]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=test_generator,
    validation_steps=50)

