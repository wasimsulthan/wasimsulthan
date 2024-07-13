#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import pickle, joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


# In[2]:


adv_data=pd.read_csv("C:\\Program Files\\datasets\\advertising.csv")


# In[3]:


user="root"
pw="root"
db="logistic"
from sqlalchemy import create_engine
engine= create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
adv_data.to_sql("logistic",con= engine , if_exists='replace',chunksize=1000,index=False)
sql="select * from logistic"
df=pd.read_sql_query(sql,engine)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.Ad_Topic_Line.unique()
df.Ad_Topic_Line.value_counts()


# In[9]:


df.City.unique()
df.City.value_counts()


# In[10]:


df.Country.unique()
df.Country.value_counts()


# In[11]:


df.Timestamp.unique()
df.Timestamp.value_counts()


# In[12]:


import sweetviz as sv
eda= sv.analyze(df)
eda.show_html("eda_report.html")
eda


# In[13]:


x=df[["Daily_Time_ Spent _on_Site","Age","Area_Income","Daily Internet Usage"]]
y=df[["Clicked_on_Ad"]]


# In[14]:


numerical_features1=x.select_dtypes(include=['int64']).columns
numerical_features2=x.select_dtypes(include=['float64']).columns


# In[15]:


print(numerical_features1)
print(numerical_features2)


# In[38]:


pipeline1=Pipeline([("impute",SimpleImputer(strategy='most_frequent'))])
pipeline1


# In[39]:


pipeline2=Pipeline([("impute",SimpleImputer(strategy='mean')),("scale",StandardScaler())])
pipeline2


# In[40]:


processed=ColumnTransformer([('mode',pipeline1,numerical_features1),
                            ('mean',pipeline2,numerical_features2)])
processed


# In[41]:


cleaned=pd.DataFrame(processed.fit_transform(x),columns=x.columns)


# In[42]:


winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=["Daily_Time_Spent_on_Site", "Age", "Area_Income", "Daily_Internet_Usage"])


# In[43]:


numerical_features = x.select_dtypes(exclude=['object']).columns


# In[44]:


outliers_pipeline=Pipeline(steps=[("winsor",winsor)])
outliers_pipeline


# In[45]:


processed2=ColumnTransformer([('wins',outliers_pipeline,numerical_features)],remainder = 'passthrough')


# In[46]:


x2=pd.DataFrame(processed.fit_transform(cleaned),columns=cleaned.columns)


# In[47]:


x2.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x2,y,test_size=0.2,random_state=0,stratify=y)


# In[49]:


logisticmodel = sm.Logit(y_train, x_train).fit()


# In[50]:


y_pred_train = logisticmodel.predict(x_train)  
y_pred_train


# In[52]:


y_train["pred"] = np.zeros(800)


# In[56]:


optimal_threshold=0.5


# In[57]:


y_train.loc[y_pred_train > optimal_threshold, "pred"] = 1


# In[61]:


print(y_train.columns)


# In[64]:





# In[66]:


auc = metrics.roc_auc_score(y_train['Clicked_on_Ad'], y_train['pred'])
print("Area under the ROC curve: %f" % auc)


# In[68]:


threshold = 0.5
y_pred_labels = (y_train["pred"] > threshold).astype(int)
classification_train = classification_report(y_train["Clicked_on_Ad"], y_pred_labels)
print(classification_train)


# In[69]:


confusion_matrix(y_train["pred"], y_train["Clicked_on_Ad"])


# In[70]:


print('Train accuracy = ', accuracy_score(y_train["pred"], y_train["Clicked_on_Ad"]))


# In[71]:


y_pred_test = logisticmodel.predict(x_test)  
y_pred_test


# In[73]:


y_test["y_pred_test"] = np.zeros(200)


# In[74]:


y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1


# In[76]:


classification1 = classification_report(y_test["y_pred_test"], y_test["Clicked_on_Ad"])
print(classification1)


# In[77]:


confusion_matrix(y_test["y_pred_test"], y_test["Clicked_on_Ad"])


# In[78]:


print('Test accuracy = ', accuracy_score(y_test["y_pred_test"], y_test["Clicked_on_Ad"]))


# In[ ]:




