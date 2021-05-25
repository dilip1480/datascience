#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[77]:



import pandas as pd
from word2number import w2n
from sklearn import linear_model
def change(l):
    if l=="":
        return 0
    else:
        d=w2n.word_to_num(l)
        return d
df=pd.read_csv("hiring.csv",converters={"experience":change})
d=df["test_score(out of 10)"].median()
df["test_score(out of 10)"].fillna(d,inplace=True)
reg=linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)',
       ]],df["salary($)"])
reg.predict([[2,9,6]])
reg.predict([[12,10,10]])

