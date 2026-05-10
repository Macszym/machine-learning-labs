#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# # Przygotowanie danych

# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[3]:


from sklearn.datasets import load_iris
data_iris = load_iris()


# In[4]:


X_bc = data_breast_cancer.data
y_bc = data_breast_cancer.target

X_iris = data_iris.data
y_iris = data_iris.target


# # Zadanie 1 + 2

# In[5]:


from sklearn.decomposition import PCA


# In[6]:


pca_bc_ns = PCA(n_components=0.9)
pca_iris_ns = PCA(n_components=0.9)


# In[7]:


X_bc_ns = pca_bc_ns.fit_transform(X_bc)
X_iris_ns = pca_iris_ns.fit_transform(X_iris)


# In[8]:


scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(X_bc)

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)


# In[9]:


pca_bc_s = PCA(n_components=0.9)
X_bc_s = pca_bc_s.fit_transform(X_bc_scaled)

pca_iris_s = PCA(n_components=0.9)
X_iris_s = pca_iris_s.fit_transform(X_iris_scaled)


# In[10]:


X_bc_s.shape


# In[11]:


X_bc.shape


# In[12]:


X_iris_s.shape


# In[13]:


X_iris.shape


# # Zadanie 3

# In[14]:


pca_bc_s_list = list(pca_bc_s.explained_variance_ratio_)
with open('pca_bc.pkl', 'wb') as f:
    pickle.dump(pca_bc_s_list, f)

pca_iris_s_list = list(pca_iris_s.explained_variance_ratio_)
with open('pca_ir.pkl', 'wb') as f:
    pickle.dump(pca_iris_s_list, f)


# # Zadanie 4

# In[15]:


weights_bc = np.abs(pca_bc_s.components_ * pca_bc_s.explained_variance_ratio_[:, np.newaxis])
indices_bc = np.argsort(np.max(weights_bc, axis=0))[::-1]

weights_iris = np.abs(pca_iris_s.components_ * pca_iris_s.explained_variance_ratio_[:, np.newaxis])
indices_iris = np.argsort(np.max(weights_iris, axis=0))[::-1]


# In[16]:


with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(list(indices_bc), f)

with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(list(indices_iris), f)

