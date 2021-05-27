#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.chdir("D:\\")
import pandas as pd
d=pd.read_csv("HR_comma_sep.csv")
df=d.copy()
df


# In[6]:


first=df.drop(['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Department'],axis="columns")


# In[7]:


first


# In[9]:


dummies=pd.get_dummies(first.salary)
dummies


# In[17]:


second=[((first.left)*(dummies.high)).sum(),((first.left)*(dummies.low)).sum(),((first.left)*(dummies.medium)).sum()]
second


# In[18]:


thrid=df.drop(['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years','salary'],axis="columns")


# In[19]:


dummies1=pd.get_dummies(thrid.Department)


# In[20]:


dummies1


# In[22]:


four=[((thrid.left)*(dummies1.IT)).sum(),((thrid.left)*(dummies1.RandD)).sum(),((thrid.left)*(dummies1.accounting)).sum(),((thrid.left)*(dummies1.hr)).sum(),((thrid.left)*(dummies1.management)).sum(),((thrid.left)*(dummies1.marketing)).sum(),((thrid.left)*(dummies1.product_mng)).sum(),((thrid.left)*(dummies1.sales)).sum(),((thrid.left)*(dummies1.support)).sum(),((thrid.left)*(dummies1.technical)).sum()]


# In[23]:


four


# In[33]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
p=["high salary","low salary","medium salary"]
q=["IT","RandD","accounting","hr","management","marketing","product_mng","sales","support","technical"]
plt.pie(second,labels=p,autopct="%0.2f%%")


# In[32]:


plt.pie(four,labels=q,autopct="%0.2f%%")


# In[77]:


final=df.drop(["salary","Department","left"],axis="columns")
final1=dummies.drop("medium",axis="columns")
final_output=pd.concat([final1,final],axis="columns")
final_output


# In[78]:


from sklearn import linear_model
reg=linear_model.LogisticRegression()
reg.fit(final_output,df.left)

reg.score(final_output,df.left)
reg.predict([[0,1,0.38,0.53,2,157,3,0,0]])


# In[ ]:




