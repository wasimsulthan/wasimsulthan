#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import linear_model,neighbors, naive_bayes, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle


# In[2]:


data=pd.read_csv("C:\Program Files\datasets\ClothCompany_Data.csv")


# In[3]:


from sqlalchemy import create_engine
user='root'
pw='root'
db='cloth'
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[4]:


data.to_sql("ensemble",con=engine,if_exists='replace',chunksize=1000,index=False)


# In[5]:


sql="select * from ensemble"
df=pd.read_sql_query(sql,engine)


# In[6]:


df.head()


# In[7]:


df.Sales.unique()
df.Sales.value_counts()


# In[8]:


df.CompPrice.unique()
df.CompPrice.value_counts()


# In[9]:


df.ShelveLoc.unique()
df.ShelveLoc.value_counts()


# In[10]:


df.Urban.unique()
df.Urban.value_counts()


# In[11]:


df.US.unique()
df.US.value_counts()


# In[12]:


numerical=df.select_dtypes(exclude=['object']).columns


# In[13]:


categorical=df.select_dtypes(include=['object']).columns


# In[14]:


numerical


# In[15]:


categorical


# In[16]:


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
numerical_pipeline=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
numerical_pipeline


# In[17]:


from sklearn.preprocessing import OneHotEncoder
categorical_pipeline=make_pipeline(OneHotEncoder())
categorical_pipeline


# In[18]:


df.describe()


# In[19]:


from sklearn.compose import ColumnTransformer 
processed_pipeline=ColumnTransformer([('numerical',numerical_pipeline,numerical),
                                     ('categorical',categorical_pipeline,categorical)])


# In[27]:


cleaned=pd.DataFrame(processed_pipeline.fit_transform(df))


# In[28]:


columns_after_transform = list(numerical) + list(processed_pipeline.transformers_[1][1].named_steps['onehotencoder'].get_feature_names_out(categorical))


# In[29]:


cleaned.columns = columns_after_transform


# In[30]:


cleaned.head()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=cleaned.select_dtypes(exclude=['object']))
plt.title('Box Plot of Numerical Columns')
plt.show()


# In[32]:


get_ipython().system('pip install feature_engine')


# In[33]:


cleaned.columns


# In[34]:


from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                 tail='both',
                 fold=1.5,
                 variables=(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education']))


# In[35]:


outliers=winsor.fit(cleaned[['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education']])


# In[36]:


cleaned[['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education']]=outliers.transform(cleaned[['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education']])


# In[37]:


sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=cleaned.select_dtypes(exclude=['object']))
plt.title('Box Plot of Numerical Columns')
plt.show()


# In[38]:


import joblib
joblib.dump(cleaned,'cleaned_data')


# In[39]:


import os
os.getcwd()


# In[40]:


cleaned['SalesCategory'] = pd.cut(cleaned['Sales'], bins=[-float('inf'), 5, 10, float('inf')],
                                  labels=['Low', 'Medium', 'High'])


# In[41]:


cleaned.head()


# In[43]:


X = cleaned.drop(['Sales', 'SalesCategory'], axis=1)
y = cleaned['SalesCategory']


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)


# In[47]:


ensemble_model = VotingClassifier(estimators=[('decision_tree', decision_tree_model), 
                                              ('random_forest', random_forest_model),
                                              ('adaboost', adaboost_model)],
                                  voting='hard')


# In[48]:


ensemble_model.fit(X_train, y_train)


# In[49]:


ensemble_predictions = ensemble_model.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report
print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))
print("Classification Report for Ensemble Model:")
print(classification_report(y_test, ensemble_predictions))


# In[ ]:




