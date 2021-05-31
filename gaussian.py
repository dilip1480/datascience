#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[6]:


from sklearn.datasets import load_wine
wines=load_wine()
dir(wines)


# In[10]:


wines.feature_names


# In[57]:


import pandas as pd
wine=pd.DataFrame(wines.data,columns=['alcohol',
 'malic_acid',
 'ash',
 'alcalinity_of_ash',
 'magnesium',
 'total_phenols',
 'flavanoids',
 'nonflavanoid_phenols',
 'proanthocyanins',
 'color_intensity',
 'hue',
 'od280/od315_of_diluted_wines',
 'proline'])
final=pd.DataFrame(wines.target,columns=['target'])
winee=pd.concat([wine,final],axis="columns")
winee
from sklearn.model_selection import cross_val_score
cross_val_score(MultinomialNB(),wine,final)


# In[58]:


cross_val_score(GaussianNB(),wine,final)

