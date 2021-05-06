
# coding: utf-8

# In[1]:


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


# (1) 读取图像文件。
# 
# (2) 将 JPEG 文件解码为 RGB 像素网格。
# 
# (3) 将这些像素网格转换为浮点数张量。
# 
# (4) 将像素值（0~255 范围内）缩放到 [0, 1] 区间（正如你所知，神经网络喜欢处理较小的输入值）。

# In[4]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)                        
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


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


for im_batch in train_generator:
    for im in im_batch:
        plt.imshow(im[0])
        break
    break


# 创建模型

# In[9]:


model = keras.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200,200,3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[10]:


model.summary()


# In[11]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=test_generator,
    validation_steps=50)


# In[ ]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

