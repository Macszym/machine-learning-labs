#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# # Przygotowanie danych

# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# # Zadanie 1

# In[3]:


from sklearn.cluster import KMeans


# In[4]:


k_list = [8, 9, 10, 11, 12]


# In[5]:


k_means_models = []
for k in k_list:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit_predict(X)
    k_means_models.append(kmeans)


# In[6]:


k_means_models


# # Zadanie 2

# In[7]:


from sklearn.metrics import silhouette_score


# In[8]:


kmeans_sil = [silhouette_score(X, model.labels_) for model in k_means_models]


# In[9]:


with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(kmeans_sil, f)


# # Zadanie 3

# In[10]:


kmeans_sil


# In[11]:


# Nie


# # Zadanie 4

# In[12]:


from sklearn.metrics import confusion_matrix


# In[13]:


conf_matrix = confusion_matrix(y, k_means_models[2].labels_)


# In[14]:


print(conf_matrix)


# # Zadanie 5

# In[15]:


indeksy = set()
max_indexes = np.argmax(conf_matrix, axis=1)

for index in max_indexes:
    indeksy.add(index)

max_indexes = list(indeksy)
max_indexes.sort()


# In[16]:


with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(max_indexes, f)


# # Zadanie 6

# In[17]:


dsts = []

for i in range(300):
    for j in range(X.shape[0]):
        distance = np.linalg.norm(X[i] - X[j])
        if distance != 0:
            dsts.append(distance)



# In[18]:


dsts.sort()


# In[19]:


dsts[:10]


# In[20]:


with open('dist.pkl', 'wb') as f:
    pickle.dump(dsts[:10], f)


# # Zadanie 7

# In[21]:


from sklearn.cluster import DBSCAN


# In[22]:


ls = np.array(dsts[:3])


# In[23]:


s = ls.mean()


# In[24]:


epsilon = s
models = []

while(epsilon <= 1.1*s):
    model = DBSCAN(eps=epsilon)
    model.fit(X)
    models.append(model)
    epsilon += 0.04*s


# In[25]:


lengths = []
for i in range(len(models)):
    lengths.append(len(np.unique(models[i].labels_)))


# In[26]:


with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(lengths, f)

