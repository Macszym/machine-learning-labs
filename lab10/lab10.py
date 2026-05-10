#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# # Zadanie 1 - Pobieranie danych

# In[2]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# # Zadanie 2 -  Przeszukiwanie przestrzeni hiperparametrów przy pomocy scikit-learn

# In[3]:


from scipy.stats import reciprocal


# In[4]:


n_hidden_list = [0, 1, 2, 3]
n_neurons_list = list(range(1, 101))
learning_rate_list = reciprocal(3e-4, 3e-2).rvs(1000).tolist()
optimizer_list = ['adam', 'sgd', 'nesterov']


# In[5]:


param_distribs = {
"model__n_hidden": n_hidden_list,
"model__n_neurons": n_neurons_list,
"model__learning_rate": learning_rate_list,
"model__optimizer": optimizer_list
}


# In[6]:


def build_model(n_hidden=1, n_neurons=30, optimizer='sgd', learning_rate=3e-3):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=[8]))

    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))

    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)

    model.compile(loss="mse", optimizer=optimizer)

    return model


# In[7]:


# model_test_1 = build_model(learning_rate=1e-6)
# model_test_2 = build_model(learning_rate=1e-5)
# model_test_3 = build_model(learning_rate=1e-4)
# model_test_4 = build_model(learning_rate=1e-3)
# model_test_5 = build_model(learning_rate=1e-2)
# model_test_6 = build_model(learning_rate=1e-1)


# In[8]:


# h1 = model_test_1.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))
# h2 = model_test_2.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))
# h3 = model_test_3.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))
# h4 = model_test_4.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))
# h5 = model_test_5.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))
# h6 = model_test_6.fit(X_train, y_train, epochs=5,
#                  validation_data=(X_valid, y_valid))


# In[9]:


# import matplotlib.pyplot as plt


# In[10]:


# h = [h1, h2, h3, h4, h5, h6]
# for i, h_i in enumerate(h):
#     plt.plot(h_i.history['loss'], label=f"Training loss 1e-{i+1}")
#     plt.legend()


# In[11]:


import scikeras
from scikeras.wrappers import KerasRegressor


# In[12]:


es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)


# In[13]:


keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[14]:


from sklearn.model_selection import RandomizedSearchCV


# In[15]:


rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                   n_iter=5,
                                   cv=3,
                                   verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid,y_valid), verbose=0)


# In[16]:


rnd_search_cv.best_params_


# In[17]:


import pickle

with open('rnd_search_params.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv.best_params_, f)

with open('rnd_search_scikeras.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv, f)


# # Zadanie 3 - Przeszukiwanie przestrzeni hiperparametrów przy pomocy Keras Tuner

# In[55]:


import keras_tuner as kt

def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "nesterov"])

    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[8]))

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(1))

    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return model


# In[56]:


random_search_tuner = kt.RandomSearch(build_model_kt, objective="val_loss", max_trials=10,
                                      overwrite=True, directory="my_california_housing",
                                      project_name="my_rnd_search", seed=42)


# In[57]:


import os
root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)


# In[58]:


random_search_tuner.search(X_train, y_train, epochs=100,
                           validation_data=(X_valid, y_valid),
                           callbacks=[es, tb])


# In[59]:


best_model = random_search_tuner.get_best_models(1)[0]


# In[60]:


best_hps = random_search_tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.values


# In[61]:


with open('kt_search_params.pkl', 'wb') as f:
    pickle.dump(best_hps.values, f)


# In[25]:


best_model.save('kt_best_model.keras')


# In[53]:


print(best_model.metrics_names)


# In[ ]:




