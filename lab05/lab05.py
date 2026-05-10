#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn import datasets


# # Przygotowanie danych

# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[3]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# # Klasyfikacja DT

# In[4]:


from sklearn.tree import DecisionTreeClassifier


# ## GridSearchCV

# In[5]:


X_bc = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y_bc = data_breast_cancer.target


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(X_bc, y_bc, test_size = 0.2, random_state = 42)


# In[8]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error as MSE


# In[9]:


tree_clf = DecisionTreeClassifier(random_state=42)


# In[10]:


param_grid = {'max_depth': range(1, 15)}


# In[11]:


search = GridSearchCV(estimator=tree_clf, param_grid=param_grid, scoring='f1', cv=5)
search.fit(X_bc, y_bc)


# In[12]:


search.best_params_


# ## Trening

# In[13]:


clf = DecisionTreeClassifier(max_depth=search.best_params_['max_depth'], random_state=42)


# In[14]:


clf.fit(X_bc_train, y_bc_train)


# ## Graf

# In[15]:


from sklearn import tree


# In[16]:


dot_data = tree.export_graphviz(clf,
                                feature_names=['mean texture', 'mean symmetry'],
                                class_names=['Malignant', 'Benign'],
                                rounded=True,
                                filled=True)


# In[17]:


import graphviz


# In[18]:


graph = graphviz.Source(dot_data) 


# In[19]:


graph


# In[20]:


graph.render("bc", format='png')


# ## Wyniki

# In[21]:


results = [clf.max_depth,
           f1_score(y_bc_train, clf.predict(X_bc_train)),
           f1_score(y_bc_test, clf.predict(X_bc_test)),
           accuracy_score(y_bc_train, clf.predict(X_bc_train)),
           accuracy_score(y_bc_test, clf.predict(X_bc_test))]


# In[22]:


with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(results, f)


# # Regresja DT

# In[23]:


from sklearn.tree import DecisionTreeRegressor


# ## GridSearchCV

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=42)


# In[25]:


param_grid = {'max_depth': range(1, 100)}


# In[26]:


tree_reg = DecisionTreeRegressor(random_state=42)


# In[27]:


search = GridSearchCV(estimator=tree_reg, param_grid=param_grid, scoring='neg_mean_squared_error')
search.fit(X.reshape(-1,1), y)


# ## Treni

# In[28]:


reg = DecisionTreeRegressor(max_depth=search.best_params_['max_depth'], random_state=42)


# In[29]:


reg.fit(X_train.values.reshape(-1,1), y_train)


# ## Graf

# In[30]:


dot_data = tree.export_graphviz(reg,
                                rounded=True,
                                filled=True)


# In[31]:


graph = graphviz.Source(dot_data) 


# In[32]:


graph


# In[33]:


graph.render('reg', format='png')


# ## Wyniki

# In[34]:


results = [reg.max_depth,
           MSE(y_train, reg.predict(X_train.values.reshape(-1,1))),
           MSE(y_test, reg.predict(X_test.values.reshape(-1,1)))]


# In[35]:


with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(results, f)

