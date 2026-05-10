#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[2]:


mnist = fetch_openml('mnist_784', version=1)


# In[3]:


mnist.data


# In[4]:


mnist.target


# In[5]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[6]:


pixels = np.array(mnist.data.loc[42]).reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()


# In[7]:


X = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target)


# In[8]:


sorted_y = y.sort_values('class', ascending=True)


# In[9]:


sorted_y


# In[10]:


sorted_X = X.reindex(sorted_y.index)


# In[11]:


sorted_X


# In[12]:


y.shape, X.shape


# In[13]:


X_train, X_test = sorted_X[:56000], sorted_X[56000:]
y_train, y_test = sorted_y[:56000], sorted_y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[14]:


y_train['class'].unique()


# In[15]:


y_test['class'].unique()


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(sorted_X, sorted_y, test_size = 0.2, random_state = 42)


# In[17]:


y_train['class'].unique(), y_test['class'].unique()


# In[18]:


y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)


# In[19]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
y_train_0 = y_train_0['class']
y_test_0 = y_test_0['class']


# In[20]:


from sklearn.linear_model import SGDClassifier


# In[21]:


sgd_clf_0 = SGDClassifier(random_state=42)
sgd_clf_0.fit(X_train, y_train_0)


# In[22]:


y_train_pred = sgd_clf_0.predict(X_train)
y_test_pred = sgd_clf_0.predict(X_test)


# In[23]:


acc_train_0 = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test_0 = sum(y_test_pred == y_test_0)/len(y_test_0)


# In[24]:


print(acc_train_0, acc_test_0)


# In[25]:


import pickle


# In[26]:


with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump([acc_train_0, acc_test_0], f)


# In[27]:


from sklearn.model_selection import cross_val_score


# In[28]:


score = cross_val_score(sgd_clf_0, X_train, y_train_0,
                        cv=3, scoring='accuracy',
                        n_jobs=-1)


# In[29]:


score


# In[30]:


with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(score, f)


# In[31]:


y_train = y_train['class']
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)


# In[32]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


# In[33]:


y_train_pred = cross_val_predict(sgd_clf, X_train,
                                 y_train, cv=3, n_jobs=-1)


# In[34]:


conf_mx = confusion_matrix(y_train, y_train_pred)


# In[35]:


print(conf_mx)


# In[36]:


with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(conf_mx, f)

