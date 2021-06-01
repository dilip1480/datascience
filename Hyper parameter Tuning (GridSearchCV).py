#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[2]:


from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


# In[21]:


digit=load_digits()
dir(digit)


# In[27]:


import pandas as pd
digits=pd.DataFrame(digit.data,columns=digit.feature_names)
targets=pd.DataFrame(digit.target,columns=["target"])
digits=pd.concat([digits,target],axis="columns")
digits


# In[94]:


model_path={
    "SVC":{
        "model":SVC(gamma="auto"),
        "par":{
             "C":[1,10,20,30],
             "kernel":("rbf","linear")}
    },
     "LogisticRegression":{
         "model":LogisticRegression(),
         "par":{
             "C":[1,10,20,30],
             "penalty":('l1', 'l2', 'elasticnet', 'none')
         }
     },
    "DecisionTreeClassifier":{
        "model":DecisionTreeClassifier(),
        "par":{
            "criterion":("gini", "entropy")
        }
    },
    "MultinomialNB":{
        "model":MultinomialNB(),
        "par":{
            "alpha":[0,0.5,1]
        }
        
    },
    "GaussianNB":{
        "model":GaussianNB()
    }
}


# In[50]:


model_path.items()


# In[96]:


score=[]
for model,mp in model_path.items():
    clf=GridSearchCV(mp["model"],mp["par"],cv=5, return_train_score=False)
    clf.fit(digits,targets)
    score.append({
        "model":model,
        "best_score":clf.best_score_,
        "best par":clf.best_params_
    })


# In[97]:


df1=pd.DataFrame(score)
df1

