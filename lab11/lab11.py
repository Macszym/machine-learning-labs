#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# # Ładowanie danych

# In[2]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load("tf_flowers",
                                                               split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
                                                               as_supervised=True,
                                                               with_info=True)


# In[3]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[4]:


info


# In[5]:


plt.figure(figsize=(12, 8))

index = 0
sample_images = train_set_raw.take(9)

for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

plt.show(block=False)


# # Budujemy prostą sieć CNN

# ## Przygotowanie danych

# In[6]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# In[7]:


batch_size = 32

train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[8]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)

for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

plt.show()


# ## Budowa sieci

# In[9]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=[224, 224, 3]),
    tf.keras.layers.Rescaling(scale=1./255),

    tf.keras.layers.Conv2D(32, (5, 5), padding='same', kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation('selu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation('selu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation('selu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.GlobalMaxPooling2D(),

    tf.keras.layers.Dense(64, kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation('selu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation('selu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(n_classes, activation='softmax')
])


# In[10]:


model.summary()


# In[11]:


model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


# In[12]:


history = model.fit(
    train_set,
    batch_size=batch_size,
    epochs=10,
    validation_data=valid_set
)


# In[13]:


train_loss, train_acc = model.evaluate(train_set)
valid_loss, valid_acc = model.evaluate(valid_set)
test_loss, test_acc = model.evaluate(test_set)


# In[14]:


acc_tuple = (train_acc, valid_acc, test_acc)


# In[15]:


acc_tuple


# In[16]:


import pickle

with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump(acc_tuple, f)


# In[17]:


model.save('simple_cnn_flowers.keras')


# # Uczenie transferowe

# ## Przygotowanie danych

# In[18]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[19]:


train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[20]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)

for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

plt.show()


# ## Budowa sieci

# In[21]:


base_model = tf.keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False)


# In[23]:


inputs = base_model.input
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# In[24]:


for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[25]:


# model.summary()


# In[26]:


h1 = model.fit(
    train_set,
    batch_size=batch_size,
    epochs=5,
    validation_data=valid_set
)


# In[27]:


for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[28]:


h2 = model.fit(
    train_set,
    batch_size=batch_size,
    epochs=5,
    validation_data=valid_set
)


# In[29]:


train_loss, train_acc = model.evaluate(train_set)
valid_loss, valid_acc = model.evaluate(valid_set)
test_loss, test_acc = model.evaluate(test_set)


# In[30]:


acc_tuple = (train_acc, valid_acc, test_acc)


# In[31]:


acc_tuple


# In[32]:


with open('xception_acc.pkl', 'wb') as f:
    pickle.dump(acc_tuple, f)


# In[33]:


model.save('xception_flowers.keras')

