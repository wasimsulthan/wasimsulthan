#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sidetable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib
import pickle
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine


# In[4]:


data=pd.read_csv("C:\\Program Files\\datasets\\50_Startups.csv")
user='root'
pw='root'
db='lassoandridge'
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
data.to_sql("lasso",con=engine, if_exists='replace', chunksize=1000, index=False)
sql="select * from lasso"
df=pd.read_sql_query(sql,engine)


# In[5]:


df.head()


# In[6]:


df.State.unique()
df.State.value_counts()


# In[8]:


from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()
df['State']=label_encode.fit_transform(df['State'])


# In[9]:


df.head()


# In[12]:


numerical_features= df.select_dtypes(exclude=['object']).columns
numerical_features


# In[15]:


from sklearn.preprocessing import StandardScaler
pipeline=Pipeline([('impute',SimpleImputer(strategy='mean')),('scale',StandardScaler())])
pipeline


# In[18]:


processed=ColumnTransformer([('impute',pipeline,numerical_features)])


# In[19]:


cleaned=pd.DataFrame(processed.fit_transform(df),columns=df.columns)


# In[20]:


cleaned.head()


# In[26]:


winsor=Winsorizer(capping_method='iqr',
                 tail='both',
                 fold=1.5,
                 variables = ['R&D Spend','Administration','Marketing Spend','State','Profit'])


# In[27]:


outlier_pipeline = Pipeline([('wins',winsor)])
outlier_pipeline


# In[28]:


processed2=ColumnTransformer([('wins',outlier_pipeline,numerical_features)],remainder='passthrough')


# In[29]:


x2=pd.DataFrame(processed2.fit_transform(cleaned),columns= cleaned.columns)


# In[30]:


x2.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()


# In[33]:


x=df[["R&D Spend","Administration","Marketing Spend","State"]]
y=df[["Profit"]]


# In[45]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13)
lasso.fit(x2, y)


# In[46]:


lasso.intercept_
lasso.coef_


# In[47]:


plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(x2.columns))


# In[48]:


pred_lasso = lasso.predict(x2)


# In[51]:


s1 = lasso.score(x2, y.Profit)
s1


# In[53]:


np.sqrt(np.mean((pred_lasso - np.array(y['Profit']))**2))


# In[54]:


lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5 ,10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(x2, y.Profit)


# In[55]:


lasso_reg.best_params_
lasso_reg.best_score_


# In[58]:


lasso_pred = lasso_reg.predict(x2)
lasso_pred


# In[59]:


s2 = lasso_reg.score(x2, y.Profit)
s2


# In[60]:


np.sqrt(np.mean((lasso_pred - np.array(y.Profit))**2))


# In[61]:


from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.13)
rm.fit(x2 , y)


# In[62]:


rm.intercept_
rm.coef_
result = rm.coef_.flatten()
result


# In[63]:


plt.bar(height = pd.Series(result), x = pd.Series(x2.columns))


# In[64]:


rm.alpha


# In[67]:


pred_rm = rm.predict(x2)
pred_rm


# In[69]:


s3 = rm.score(x2, y.Profit)
s3


# In[70]:


np.sqrt(np.mean((pred_rm - np.array(y['Profit']))**2))


# In[71]:


ridge = Ridge()
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(x2, y.Profit)


# In[72]:


ridge_reg.best_params_
ridge_reg.best_score_


# In[73]:


ridge_pred = ridge_reg.predict(x2)


# In[75]:


s4 = ridge_reg.score(x2, y.Profit)
s4


# In[76]:


np.sqrt(np.mean((ridge_pred - np.array(y.Profit))**2))


# In[77]:


from sklearn.linear_model import ElasticNet 
enet = ElasticNet(alpha = 0.13)
enet.fit(x2, y.Profit)


# In[78]:


enet.intercept_
enet.coef_


# In[79]:


plt.bar(height = pd.Series(enet.coef_), x = pd.Series(x2.columns))


# In[80]:


pred_enet = enet.predict(x2)


# In[82]:


s5 = enet.score(x2, y.Profit)
s5


# In[83]:


np.sqrt(np.mean((pred_enet - np.array(y.Profit))**2))


# In[84]:


enet = ElasticNet()


# In[85]:


enet_reg = GridSearchCV(enet, parameters, scoring = 'r2', cv = 5)
enet_reg.fit(x2, y.Profit)


# In[86]:


enet_reg.best_params_
enet_reg.best_score_


# In[87]:


enet_pred = enet_reg.predict(x2)


# In[88]:


s6 = enet_reg.score(x2, y.Profit)
s6


# In[89]:


np.sqrt(np.mean((enet_pred - np.array(y.Profit))**2))


# In[90]:


scores_all = pd.DataFrame({'models':['Lasso', 'Ridge', 'Elasticnet', 'Grid_lasso', 'Grid_ridge', 'Grid_elasticnet'], 'Scores':[s1, s2, s3, s4, s5, s6]})
scores_all


# In[91]:


finalgrid = enet_reg.best_estimator_
finalgrid


# In[92]:


pickle.dump(finalgrid, open('grid_elasticnet.pkl', 'wb'))


# In[ ]:





# In[ ]:




