#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install sidetable')
import sidetable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sqlalchemy import create_engine


# In[6]:


data=pd.read_csv("C:\\Program Files\\datasets\\50_Startups.csv")
user="root"
pw="root"
db="multilinearregression"
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')


# In[8]:


data.to_sql("mlr", con=engine ,if_exists='replace',chunksize=1000,index=False)
sql='select * from mlr'
df=pd.read_sql_query(sql,engine)


# In[9]:


df.head()


# In[10]:


df.shape


# In[12]:


df.State.unique()
df.State.value_counts()


# In[13]:


df.isnull().sum()


# In[15]:


data.info()


# In[18]:


numerical=df.select_dtypes(exclude=['object']).columns
numerical


# In[21]:


categorical=df.select_dtypes(include=['object']).columns
categorical


# In[26]:


num_pipeline=Pipeline([("impute",SimpleImputer(strategy='mean')),('scale',MinMaxScaler())])
num_pipeline


# In[ ]:


cat_pipeline=Pipeline()


# In[42]:


processed = ColumnTransformer([("numerical",num_pipeline,numerical),
                              ])
processed


# In[51]:


imputed = processed.fit_transform(df[numerical])


# In[52]:


joblib.dump(imputed,'processed')


# In[55]:


cleaned=pd.DataFrame(imputed,columns=numerical )


# In[56]:


cleaned.head()


# In[58]:


cleaned.info()


# In[60]:


cleaned.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 


# In[61]:


sns.pairplot(data)


# In[62]:


orig_df_cor = data.corr()
orig_df_cor


# In[63]:


dataplot = sns.heatmap(orig_df_cor, annot = True, cmap = "YlGnBu")


# In[64]:


mask = np.triu(np.ones_like(orig_df_cor, dtype = bool))
sns.heatmap(orig_df_cor, annot = True, mask = mask, vmin = -1, vmax = 1)
plt.title('Correlation Coefficient Of Predictors')
plt.show()


# In[67]:


y=cleaned['Profit']
x=cleaned.drop('Profit',axis=1)


# In[68]:


basemodel = sm.OLS(y,x).fit()
basemodel.summary()


# In[69]:


vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif


# In[71]:


sm.graphics.influence_plot(basemodel)


# In[72]:


X_train, X_test, Y_train, Y_test = train_test_split(cleaned, y, 
                                                    test_size = 0.2, random_state = 0) 


# In[73]:


test_model = sm.OLS(Y_train, X_train).fit()
test_model.summary()


# In[75]:


ytrain_pred = test_model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)
r_squared_train


# In[78]:


y_pred = test_model.predict(X_test)
y_pred


# In[79]:


r_squared = r2_score(Y_test, y_pred)
r_squared


# In[80]:


lm = LinearRegression()
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores  


# In[81]:


folds = KFold(n_splits = 5, shuffle = True, random_state = 100)


# In[82]:


hyper_params = [{'n_features_to_select': list(range(1, 5))}]


# In[83]:


lm.fit(X_train, Y_train)


# In[84]:


rfe = RFE(lm)


# In[85]:


model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring = 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score = True) 


# In[86]:


model_cv.fit(X_train, Y_train)     


# In[88]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[89]:


plt.figure(figsize = (16, 6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc = 'upper left')


# In[90]:


model_cv.best_params_

cv_lm_grid = model_cv.best_estimator_
cv_lm_grid


# In[91]:


pickle.dump(cv_lm_grid, open('mpg.pkl', 'wb'))


# In[ ]:




