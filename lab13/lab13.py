#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# # Przygotowanie danych

# In[2]:


dataset = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = dataset
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# # Autoenkoder z warstwami gęstymi

# In[3]:


encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(128, activation="selu", kernel_initializer='lecun_normal'),
    tf.keras.layers.Dense(32, activation="selu", kernel_initializer='lecun_normal'),
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="selu", kernel_initializer='lecun_normal', input_shape=[32]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid", kernel_initializer='glorot_normal'),
    tf.keras.layers.Reshape([28, 28])
])

ae = tf.keras.Sequential([encoder, decoder])


# In[4]:


ae.compile(optimizer=tf.keras.optimizers.Adam(),
           loss=tf.keras.losses.MeanSquaredError(),
           metrics=['mae'])


# In[5]:


history = ae.fit(X_train, X_train, validation_data=(X_valid, X_valid), epochs=5)


# In[6]:


baseline = ae.evaluate(X_test, X_test, return_dict=True)


# In[7]:


def plot_reconstructions(model, images=X_test, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")

plot_reconstructions(ae)


# In[8]:


ae.save('ae_stacked.keras')


# # Autoenkoder konwolucyjny

# In[9]:


conv_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    tf.keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2)
])

conv_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu", input_shape=[3, 3, 64]),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

conv_ae = tf.keras.Sequential([conv_encoder, conv_decoder])


# In[10]:


conv_ae.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mae'])


# In[11]:


conv_history = conv_ae.fit(X_train, X_train, validation_data=(X_valid, X_valid), epochs=5)


# In[12]:


conv_ae.evaluate(X_test, X_test)


# In[13]:


plot_reconstructions(conv_ae)
plt.show()


# In[14]:


conv_ae.save('ae_conv.keras')


# # Wizualizacja wyników

# In[15]:


from sklearn.manifold import TSNE

X_valid_compressed_4D = conv_encoder.predict(X_valid)
n_samples = X_valid_compressed_4D.shape[0]
X_valid_compressed_2D = X_valid_compressed_4D.reshape(n_samples, -1)
tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
X_valid_2D = tsne.fit_transform(X_valid_compressed_2D)

plt.figure(figsize=(10, 5))
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()


# In[16]:


import matplotlib as mpl

plt.figure(figsize=(10, 5))
cmap = plt.cm.tab10

Z = X_valid_2D
Z = (Z - Z.min()) / (Z.max() - Z.min()) # normalize to the 0-1 range

plt.scatter(Z[:, 0], Z[:, 1], c=y_valid, s=10, cmap=cmap)

image_positions = np.array([[1., 1.]])
for index, position in enumerate(Z):
    dist = ((position - image_positions) ** 2).sum(axis=1)
    if dist.min() > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
        mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
        position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)

plt.axis("off")
plt.show()


# # Odszumianie

# In[17]:


dropout_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.GaussianNoise(stddev=0.2),
    tf.keras.layers.Dense(128, activation="selu", kernel_initializer='lecun_normal'),
    tf.keras.layers.Dense(32, activation="selu", kernel_initializer='lecun_normal'),
])

dropout_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="selu", kernel_initializer='lecun_normal', input_shape=[32]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid", kernel_initializer='glorot_normal'),
    tf.keras.layers.Reshape([28, 28])
])

dropout_ae = tf.keras.Sequential([dropout_encoder, dropout_decoder])


# In[18]:


dropout_ae.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=['mae'])


# In[19]:


dropout_history = dropout_ae.fit(X_train, X_train, validation_data=(X_valid, X_valid), epochs=5)


# In[20]:


noise = tf.keras.layers.GaussianNoise(stddev=0.2)
plot_reconstructions(dropout_ae, noise(X_valid, training=True))


# In[21]:


dropout_ae.save('ae_denoise.keras')


# In[22]:


dropout_ae.evaluate(X_test, X_test)


# In[ ]:




