#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# # Przygotowanie danych

# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# # Ćwiczenie

# ## Zadanie 1

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer.data, data_breast_cancer.target, test_size = 0.2)


# ## Zadanie 2

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


dt_clf = DecisionTreeClassifier()
log_clf = LogisticRegression()
knn_clf = KNeighborsClassifier()


# In[6]:


from sklearn.ensemble import VotingClassifier

voting_clf_hard = VotingClassifier(estimators=[('dt', dt_clf),
                                          ('lr', log_clf),
                                          ('knn', knn_clf)],
                                   voting='hard')

voting_clf_soft = VotingClassifier(estimators=[('dt', dt_clf),
                                          ('lr', log_clf),
                                          ('knn', knn_clf)],
                                   voting='soft')


# ## Zadanie 3

# In[7]:


from sklearn.metrics import accuracy_score


# In[8]:


for clf in (dt_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train[['mean texture', 'mean symmetry']], y_train)
    y_pred = clf.predict(X_test[['mean texture', 'mean symmetry']])
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# ## Zadanie 4

# In[9]:


accuracy_list_1 = []


# In[10]:


for clf in (dt_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    y_pred_train = clf.predict(X_train[['mean texture', 'mean symmetry']])
    y_pred_test = clf.predict(X_test[['mean texture', 'mean symmetry']])
    accuracy_list_1.append((accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)))


# In[11]:


accuracy_list_1


# In[12]:


with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(accuracy_list_1, f)


# In[13]:


with open('vote.pkl', 'wb') as f:
    pickle.dump([dt_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft], f)


# ## Zadanie 5

# In[14]:


from sklearn.ensemble import BaggingClassifier


# In[15]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30)
bag_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5)

pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
pas_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False)


# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


rnd_clf = RandomForestClassifier(n_estimators=30)


# In[18]:


from sklearn.ensemble import AdaBoostClassifier


# In[19]:


ab_clf = AdaBoostClassifier(n_estimators=30)


# In[20]:


from sklearn.ensemble import GradientBoostingClassifier


# In[21]:


gb_clf = GradientBoostingClassifier(n_estimators=30)


# ## Zadanie 6

# In[22]:


accuracy_list_2 = []


# In[23]:


for clf in (bag_clf, bag_clf_half, pas_clf, pas_clf_half, rnd_clf, ab_clf, gb_clf):
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    accuracy_list_2.append((accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)))


# In[24]:


accuracy_list_2


# In[25]:


with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(accuracy_list_2, f)


# In[26]:


with open('bag.pkl', 'wb') as f:
    pickle.dump([bag_clf, bag_clf_half, pas_clf, pas_clf_half, rnd_clf, ab_clf, gb_clf], f)


# ## Zadanie 7

# In[27]:


specific_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                                 max_samples=0.5, max_features=2, bootstrap=True)


# In[28]:


specific_clf.fit(X_train, y_train)


# ## Zadanie 8

# In[29]:


with open('acc_fea.pkl', 'wb') as f:
    pickle.dump([accuracy_score(y_train, specific_clf.predict(X_train)), accuracy_score(y_test, specific_clf.predict(X_test))], f)


# In[30]:


with open('fea.pkl', 'wb') as f:
    pickle.dump([specific_clf], f)


# ## Zadanie 9

# In[31]:


estimators = specific_clf.estimators_
estimators_features = specific_clf.estimators_features_


# In[32]:


estimators_list = []


# In[33]:


for i, estimator in enumerate(estimators):
    
    selected_features = estimators_features[i]
    feature_names = X_train.columns[selected_features].tolist()

    X_train_selected = X_train.iloc[:, selected_features].values
    X_test_selected = X_test.iloc[:, selected_features].values
    
    estimators_list.append({
        'Train accuracy': accuracy_score(y_train, estimator.predict(X_train_selected)),
        'Test accuracy': accuracy_score(y_test, estimator.predict(X_test_selected)),
        'Selected features': feature_names
    })


# In[34]:


df = pd.DataFrame(estimators_list)


# In[35]:


df = df.sort_values(by=['Test accuracy', 'Train accuracy'], ascending=False)


# In[36]:


df.to_pickle('acc_fea_rank.pkl')

