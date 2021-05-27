#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.chdir("D:\\")
import pandas as pd
d=pd.read_csv("titanic.csv")
df=d.copy()
df


# In[24]:


rop=df.drop(["PassengerId","Survived","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis="columns")
rop


# In[25]:


dummies=pd.get_dummies(rop.Sex)
dummiess=dummies.drop("female",axis="columns")


# In[27]:


dropes=pd.concat([rop,dummies],axis="columns")
dropes


# In[59]:


X=dropes.drop(["Sex","female"],axis="columns")
X


# In[66]:


X.fillna({"Age":z["Age"].mean()},inplace=True)
X
y=df.Survived
y


# In[67]:


from sklearn import tree 
tre=tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)
tre.fit(X_train,y_train)
tre.score(X_test,y_test)

