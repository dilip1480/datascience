#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
os.chdir("D:\\")
from sklearn.datasets import load_digits
model=load_digits()
dir(model)


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(model.data,model.target,test_size=0.2)
from sklearn.svm import SVC
vector=SVC()
vector.fit(X_train,y_train)
vector.score(X_test,y_test)

