#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
os.chdir("D:\\")
from sklearn.datasets import load_iris
reg=load_iris()
dir(reg)


# In[43]:


from sklearn.model_selection import train_test_split
X=reg.data
y=reg.target
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)
from sklearn import linear_model
regg=linear_model.LogisticRegression()
regg.fit(X_train,y_train)
regg.score(X_test,y_test)


# In[46]:


regg.predict(X_test)


# In[47]:


reg.target_names[1]


# In[48]:


reg.feature_names[1]


# In[36]:


reg.target

