#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[5]:


from sklearn.datasets import load_iris
plant=load_iris()
dir(plant)


# In[7]:


plant.target


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(plant.data,plant.target,test_size=0.2)


# In[22]:


from sklearn.ensemble import RandomForestClassifier
dt=RandomForestClassifier(n_estimators=10)
dt.fit(X_train,y_train)
dt.score(X_test,y_test)

