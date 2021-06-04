#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[5]:


import pandas as pd
df=pd.read_csv("fe1.csv")
df


# In[11]:


df1=df[~(df.sqft/df.bhk<300)]
df1


# In[12]:


df1.price_per_sqft.describe()


# In[25]:


df2=df1[~(df1.bhk+1<=df1.bath)]
df2


# In[51]:


import numpy as np
def is_max(x):
    df_out=pd.DataFrame()
    group=x.groupby("location")
    for y,sub in group:
        m=np.mean(sub.price_per_sqft)
        std=np.std(sub.price_per_sqft)
        df3=sub[(((sub.price_per_sqft)>(m-std)) & ((sub.price_per_sqft)<=(m+std)))]
        df_out=pd.concat([df_out,df3],ignore_index=True)
    return df_out


# In[92]:


df3=is_max(df2)
df3


# In[103]:


def remove_bhk_outliers(x):
    exclude_indices=np.array([])
    for location,location_df in x.groupby("location"):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk]={
                "mean":np.mean(bhk_df.price_per_sqft),
                "std":np.std(bhk_df.price_per_sqft),
                "count":bhk_df.shape[0]
                }
            for bhk,bhk_df in location_df.groupby("bhk"):
                stats=bhk_stats.get(bhk-1)
                if stats and stats["count"]>5:
                    exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats["mean"])].index.values)
    return x.drop(exclude_indices,axis="index")


# In[105]:



df5= remove_bhk_outliers(df3)
df5


# In[106]:


df5.to_csv("Outlier_Removal.csv")

