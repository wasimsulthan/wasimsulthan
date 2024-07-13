#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
data=pd.read_csv("C:\Program Files\datasets\delivery_time.csv")


# In[2]:


from sqlalchemy import create_engine
user='root'
pw='root'
db='linearregression'
engine= create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[5]:


data.to_sql('linearregression',con =engine,if_exists='replace',chunksize=1000,index=False)
sql='select * from linearregression'
df=pd.read_sql_query(sql,engine)


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data['Delivery Time'].value_counts()


# In[9]:


data['Sorting Time'].value_counts()


# In[10]:


data.isnull().sum()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle


# In[13]:


data.plot(kind="box",subplots= True ,sharey=False ,figsize=(15,8))


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


# In[16]:


x=df["Sorting Time"]
y=df["Delivery Time"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[18]:


model=LinearRegression()
x_train_reshaped = x_train.to_numpy().reshape(-1, 1)
y_train_reshaped = y_train.to_numpy().reshape(-1, 1)
x_test_reshaped = x_test.to_numpy().reshape(-1, 1)
y_test_reshaped = y_test.to_numpy().reshape(-1, 1)


# In[19]:


model.fit(x_train_reshaped,y_train_reshaped)


# In[21]:


y_pred=model.predict(x_test_reshaped)
y_pred


# In[27]:


plt.scatter(df["Sorting Time"],df["Delivery Time"])
plt.plot(x_test_reshaped,y_pred, color = 'red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.legend(['Original data','Linear Regression Line'])
plt.show()


# In[29]:


rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[44]:


import numpy as np
sorting_time_array = np.array(x_test_reshaped).flatten()
y_pred_array = np.array(y_pred).flatten()
correlation_matrix = np.corrcoef(sorting_time_array, y_pred_array)
correlation_coefficient = correlation_matrix[0, 1]
print(f'Correlation Coefficient: {correlation_coefficient}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




