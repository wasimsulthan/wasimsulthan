#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install lifelines')
import pandas as pd 
from sqlalchemy import create_engine 
from lifelines import KaplanMeierFitter


# In[3]:


survival = pd.read_excel("C:\\Program Files\\datasets\\ECG_Surv.xlsx")
user='root'
pw='root'
db='ecg'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
survival.to_sql("surival",con = engine , if_exists='replace',chunksize=1000, index=False)
sql="select * from surival"
df= pd.read_sql_query(sql,engine)


# In[4]:


df.head()


# In[5]:


survival.survival_time_hr.describe()


# In[6]:


survival.info()


# In[8]:


time=survival.survival_time_hr


# In[9]:


event=survival.alive


# In[14]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
time = scale.fit_transform(survival.survival_time_hr.values.reshape(-1,1))
event = scale.fit_transform(survival.alive.values.reshape(-1,1))


# In[15]:


kmf = KaplanMeierFitter()
kmf.fit(time, event_observed = event)


# In[16]:


kmf.plot()


# In[17]:


survival.group.unique()
survival.group.value_counts()


# In[20]:


kmf.fit(time [survival.group == 1], survival.alive[survival.group == 1], label = '1')
ax = kmf.plot()


# In[22]:


kmf.fit(time[survival.group == 2], survival.alive[survival.group == 2], label = '2')
ax=kmf.plot()


# In[23]:


kmf.fit(time[survival.group == 3], survival.alive[survival.group == 3], label = '3')
ax=kmf.plot()


# In[ ]:




