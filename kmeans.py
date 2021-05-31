#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:\\")


# In[4]:


from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)


# In[8]:


import pandas as pd
data=pd.DataFrame(iris.data,columns=["petal_length","petal_width","sepal_length","sepal_width"])
data


# In[14]:


final_data=data.drop(["sepal_length","sepal_width"],axis="columns")
final_data


# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(final_data.petal_length,final_data.petal_width)


# In[34]:


from sklearn.cluster import KMeans
k=range(1,10)
sse=[]
for i in k:
    km=KMeans(n_clusters=i)
    km.fit(final_data[["petal_length","petal_width"]])
    sse.append(km.inertia_)
print(sse)


# In[35]:


plt.plot(k,sse)


# In[67]:


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(final_data[["petal_length","petal_width"]])
y_predicted


# In[44]:


dir(iris)


# In[68]:


final_data["clusters"]=y_predicted
final_data
df1=final_data[final_data.clusters==0]
df2=final_data[final_data.clusters==1]
df3=final_data[final_data.clusters==2]
plt.scatter(df1.petal_length,df1.petal_width)
plt.scatter(df2.petal_length,df2.petal_width)
plt.scatter(df3.petal_length,df3.petal_width)

