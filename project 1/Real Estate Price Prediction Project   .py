#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[2]:


import pandas as pd
df=pd.read_csv("Outlier_Removal.csv")
df.head()


# In[6]:


df1=df.drop(["Unnamed: 0","size","total_sqft","balcony","price_per_sqft"],axis="columns")
df1.head()


# In[8]:


dummins=pd.get_dummies(df1.location)
dummins


# In[12]:


df2=pd.concat([df1,dummins],axis="columns")
df3=df2.drop("location",axis="columns")


# In[15]:


df4=df3.drop("other",axis="columns")
df4


# In[91]:


x=df4.drop("price",axis="columns")
y=df4.price
y


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[39]:


cross_val_score(LinearRegression(),x,y,cv=10)


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
d=LinearRegression()
d.fit(X_train,y_train)
d.score(X_test,y_test)


# In[65]:


from sklearn.model_selection import GridSearchCV
model={
    "LinearRegression":{
        "models":LinearRegression(),
        "parms":{
            "fit_intercept":[True,False],
            "normalize":[True,False]
        }
    
        
    }
    
}


# In[74]:


score=[]
for kk,sub in model.items():
    gf=GridSearchCV(sub['models'],sub["parms"],cv=5,return_train_score=False)
    gf.fit(x,y)
    score.append({
        "model":kk,
        "score":gf.best_score_,
        "parms":gf.best_params_
    })
score


# In[89]:


import numpy as np
def predict_price(bath,bhk,sqft,location):
    s=np.where(x.columns==location)[0][0]
    p=np.zeros(len(x.columns))
    p[0]= bath
    p[1]= bhk
    p[2]= sqft
    p[s-1]=1
    return d.predict([p])[0]


# In[90]:


predict_price(4,4,2850,"1st Block Jayanagar")


# In[97]:


import pickle
with open('home_price.pickle','wb') as f:
    pickle.dump(d,f)


# In[98]:


import json
columns={
    "data_columns":[col.lower() for col in x.columns]
}
with open("colmns.json","w") as f:
    f.write(json.dumps(columns))

