#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import urllib.request
import tarfile
import gzip
import shutil


# In[2]:


os.makedirs('data')


# In[3]:


url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'
urllib.request.urlretrieve(url, 'data/housing.tgz')


# In[4]:


file = tarfile.open('data/housing.tgz')
file.extractall('./data')
file.close()
os.remove('data/housing.tgz')


# In[5]:


with open('./data/housing.csv', 'rb') as f_in:
    with gzip.open('./data/housing.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[6]:


import pandas as pd

df = pd.read_csv('data/housing.csv.gz')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df['ocean_proximity'].value_counts()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


df.hist(bins=50, figsize=(20,15))

plt.savefig('obraz1.png')


# In[12]:


df['ocean_proximity'].describe()


# In[13]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.1, figsize=(7,4))

plt.savefig('obraz2.png')


# In[14]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.4, figsize=(7,3), colorbar=True,
s=df["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"))

plt.savefig('obraz3.png')


# In[15]:


pd.get_dummies(df).corr()["median_house_value"].sort_values(ascending=False)


# In[16]:


pd.get_dummies(df).corr()['median_house_value'].sort_values(ascending=False).reset_index().rename(columns={"index": "atrybut", "median_house_value": "wspolczynnik_korelacji"}).to_csv("korelacja.csv", index=False)


# In[17]:


import seaborn as sns

sns.pairplot(df)


# In[18]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(pd.get_dummies(df), test_size=0.2, random_state=42)
len(train_set), len(test_set)


# In[19]:


train_set.sample(5)


# In[20]:


test_set.sample(5)


# In[21]:


train_set.corr()['median_house_value'].sort_values(ascending=False)


# In[22]:


test_set.corr()['median_house_value'].sort_values(ascending=False)


# In[23]:


((test_set.corr()['median_house_value']-train_set.corr()['median_house_value'])).sort_values(ascending=False)


# In[24]:


train_set.to_pickle('train_set.pkl')
test_set.to_pickle('test_set.pkl')

