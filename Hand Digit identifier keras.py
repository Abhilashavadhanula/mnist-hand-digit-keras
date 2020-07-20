#!/usr/bin/env python
# coding: utf-8

# In[45]:


## This is simple NN 
## Builded on mnist dataset


# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


len(X_test)


# In[5]:



X_train[0].shape


# In[6]:


X_train[0]


# In[8]:


plt.matshow(X_train[1])


# In[9]:



y_train[1]


# In[11]:


## This is used to do scaling inorder to improve the accuracy of the model


# In[12]:


X_train = X_train / 255
X_test = X_test / 255


# In[13]:


X_train[0]


# In[14]:


## This is for flatteneing the array from 2d to 1 d


# In[15]:



X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[16]:



X_train_flattened.shape


# In[17]:


X_train_flattened[0]


# In[18]:


### Now creating the Deep Learning NN (simple one output and 1 inout layer)


# In[19]:



model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[20]:


### Evalutaion of the model


# In[21]:


model.evaluate(X_test_flattened, y_test)


# In[23]:


## This is the predicted output which one got high accuracy will be seen
y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[24]:


plt.matshow(X_test[0])


# ### np.argmax finds a maximum element from an array and returns the index of it

# In[26]:


np.argmax(y_predicted[0])


# In[27]:



y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[28]:



y_predicted_labels[:5]


# In[29]:


### confusin matrix in the tensorflow


# In[30]:



cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[31]:


## Plotting the confusin matrix 


# In[33]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[36]:


### Using hidden layer
## it takes some more time compared to no hidden layer 


# In[35]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[37]:


model.evaluate(X_test_flattened,y_test)


# In[40]:


## we can obsorbe the increase in the accuracy of the model
## we can check by ploting the cm graph 


# In[39]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ## Using Flatten layer so that we don't have to call .reshape on input dataset

# In[41]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[42]:



model.evaluate(X_test,y_test)


# In[43]:


y_predicted[1]


# In[44]:


plt.matshow(X_test[1])


# In[ ]:




