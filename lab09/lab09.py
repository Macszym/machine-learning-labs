#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# # Klasyfikacja obrazów

# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[3]:


X_train = X_train / 255
X_test = X_test / 255


# In[4]:


plt.figure(figsize = (2,2))
plt.imshow(X_train[42], cmap="binary")
plt.axis('off')
plt.show()


# In[5]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "półbut", "torba", "but"]
class_names[y_train[42]]


# In[6]:


model = tf.keras.models.Sequential()


# In[7]:


model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))


# In[8]:


model.summary()


# In[9]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# In[10]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()


# In[11]:


tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[12]:


history = model.fit(X_train, y_train, epochs=20,
                    validation_split=0.1, callbacks=tensorboard_cb)


# In[13]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.figure(figsize = (2,2))
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[14]:


model.save('fashion_clf.keras')


# # Regresja

# In[15]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()


# In[16]:


X = housing.data
y = housing.target


# In[17]:


X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.1)

X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1)


# In[18]:


model_h = tf.keras.models.Sequential()


# In[19]:


normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train)


# In[20]:


model_h.add(normalizer)
model_h.add(tf.keras.layers.Dense(50, activation='relu', input_shape=[X_train.shape[1]]))
model_h.add(tf.keras.layers.Dense(50, activation='relu'))
model_h.add(tf.keras.layers.Dense(50, activation='relu'))
model_h.add(tf.keras.layers.Dense(1))


# In[21]:


model_h.compile(loss="MeanSquaredError",
              optimizer="Adam",
              metrics=["RootMeanSquaredError"])


# In[22]:


model_h.summary()


# In[23]:


es = tf.keras.callbacks.EarlyStopping(patience=5,
                                      min_delta=0.01,
                                      verbose=1)


# In[24]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[25]:


history_h = model_h.fit(X_train, y_train, epochs=200,
                        validation_data=(X_val, y_val), callbacks=[tensorboard_cb, es])


# In[26]:


model_h.save('reg_housing_1.keras')


# In[27]:


model_h_2 = tf.keras.models.Sequential()
model_h_2.add(tf.keras.layers.Dense(50, activation='relu', input_shape=[X_train.shape[1]]))
model_h_2.add(tf.keras.layers.Dense(50, activation='relu'))
model_h_2.add(tf.keras.layers.Dense(50, activation='relu'))
model_h_2.add(tf.keras.layers.Dense(1))
model_h_2.compile(loss="MeanSquaredError",
              optimizer="Adam",
              metrics=["RootMeanSquaredError"])
model_h_2.summary()


# In[28]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history_h_2 = model_h_2.fit(X_train, y_train, epochs=200,
                            validation_data=(X_val, y_val), callbacks=[tensorboard_cb, es])


# In[29]:


model_h_2.save('reg_housing_2.keras')


# In[30]:


model_h_3 = tf.keras.models.Sequential()
model_h_3.add(normalizer)
model_h_3.add(tf.keras.layers.Dense(200, activation='relu'))
model_h_3.add(tf.keras.layers.Dense(300, activation='relu'))
model_h_3.add(tf.keras.layers.Dense(100, activation='relu'))
model_h_3.add(tf.keras.layers.Dense(1))

model_h_3.compile(loss="MeanSquaredError",
                  optimizer="Adam",
                  metrics=["RootMeanSquaredError"])

model_h_3.summary()


# In[31]:


root_logdir = os.path.join(os.curdir, "housing_logs")
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history_h_3 = model_h_3.fit(X_train, y_train, epochs=200,
                            validation_data=(X_val, y_val), callbacks=[tensorboard_cb, es])


# In[32]:


model_h_3.save('reg_housing_3.keras')

