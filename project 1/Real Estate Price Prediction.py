#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.chdir("D:\\")


# In[7]:


import pandas as pd
df=pd.read_csv("cleaned.csv")
df


# In[8]:


def is_ok(x):
    try:
        z=x/x
        return x
    except:
        return NaN


# In[10]:


df.sqft.apply(lambda x:is_ok(x))


# In[15]:


df.isna().sum()


# In[14]:


df.dropna(inplace=True)
df


# In[26]:


df["price_per_sqft"]=(df.price)*100000/df.sqft
df


# In[39]:


df.location=df.location.apply(lambda x:x.strip())


# In[45]:


location_status=df.groupby("location")["location"].agg("count").sort_values(ascending=False)
location_status<10


# In[48]:


location_status_less_than_10=location_status[location_status<=10]
location_status_less_than_10


# In[51]:


df.location=df.location.apply(lambda x:"other" if x in location_status_less_than_10 else x)
df.head(20)

