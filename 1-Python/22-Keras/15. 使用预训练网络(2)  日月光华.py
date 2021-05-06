
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


# In[4]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)                        
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


covn_base = keras.applications.VGG16(weights='imagenet', 
                                     include_top=False, 
                                     input_shape=(200, 200, 3))


# In[7]:


covn_base.summary()


# In[8]:


batch_size=20


# In[9]:


def extract_features(data_generator, sample_count):
    i = 0
    features = np.zeros(shape=(sample_count, 6, 6, 512))
    labels = np.zeros(shape=(sample_count))
    for inputs_batch, labels_batch in data_generator:
        features_batch = covn_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# In[10]:


train_features, train_labels = extract_features(train_generator, 2000)
test_features, test_labels = extract_features(test_generator, 1000)


# In[23]:


model = keras.Sequential()
model.add(layers.GlobalAveragePooling2D(input_shape=(6, 6, 512)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[24]:


model.summary()


# In[25]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])


# In[26]:


history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=50,
                    validation_data=(test_features, test_labels))


# In[27]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[28]:


plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
plt.plot(history.epoch, history.history['val_loss'], 'b--', label='val_loss')
plt.legend()


# In[29]:


plt.plot(history.epoch, history.history['acc'], 'r')
plt.plot(history.epoch, history.history['val_acc'], 'b--')

