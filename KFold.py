#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
os.chdir("D:\\")
from sklearn.datasets import load_iris 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
final=load_iris()
dir(final)


# In[14]:


cross_val_score(LogisticRegression(),final.data,final.target)


# In[11]:


cross_val_score(SVC(),final.data,final.target)


# In[21]:


cross_val_score(RandomForestClassifier(n_estimators=100),final.data,final.target)

