#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree


# In[2]:


data=pd.read_csv("C:\Program Files\datasets\ClothCompany_Data.csv")


# In[3]:


from sqlalchemy import create_engine
user="root"
pw="root"
db="cloth"
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[4]:


data.to_sql("cloth",con=engine,if_exists='replace',chunksize=1000,index=False)


# In[5]:


sql="select * from cloth"
df=pd.read_sql_query(sql,engine)


# In[6]:


df.head()


# In[7]:


df.info


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


sns.pairplot(data=df, hue = 'ShelveLoc')


# In[12]:


df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)


# In[13]:


df


# In[14]:


df.info()


# In[15]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[16]:


df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})


# In[17]:


df.head()


# In[18]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']


# In[19]:


df['ShelveLoc'].unique()


# In[20]:


df.ShelveLoc.value_counts()


# In[21]:


colnames = list(df.columns)
colnames


# In[22]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[23]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[24]:


from sklearn import tree


# In[25]:


tree.plot_tree(model);


# In[26]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[27]:


preds = model.predict(x_test) 
pd.Series(preds).value_counts()


# In[28]:


preds


# In[29]:


pd.crosstab(y_test,preds)


# In[30]:


np.mean(preds==y_test)


# In[ ]:




