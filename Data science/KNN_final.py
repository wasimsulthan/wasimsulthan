#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


# In[4]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# In[6]:


from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# In[15]:


import sklearn.metrics as skmet
import pickle
from sqlalchemy import create_engine


# In[16]:


glass = pd.read_csv(r"C:\Program Files\datasets\glass.csv")


# In[22]:


user="root"
pw="root"
db="glass"
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[23]:


glass.to_sql('glass',con=engine,if_exists='replace',chunksize=1000,index=False)


# In[24]:


sql='select * from glass'
glass_df= pd.read_sql_query(sql,engine)


# In[25]:


glass_df.head()


# In[28]:


print(glass_df)


# In[30]:


glass_df.describe()


# In[38]:


glass_df_x=pd.DataFrame(glass_df.iloc[:,0:9])
glass_df_x
glass_df_z=pd.DataFrame(glass_df.iloc[:,-1])
glass_df_z


# In[39]:


glass_df.info()


# In[57]:


numeric_features=glass_df_x.select_dtypes(exclude=['object']).columns


# In[55]:


pipeline=Pipeline([('impute',SimpleImputer(strategy='mean')),('scale',MinMaxScaler())])
pipeline


# In[58]:


cleaned=pd.DataFrame(pipeline.fit_transform(glass_df_x),columns= glass_df_x.columns)


# In[59]:


import joblib
joblib.dump(cleaned,"cleaned")


# In[60]:


import os
os.getcwd()


# In[61]:


res=glass_df_x.describe()
res


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(cleaned,glass_df_z,test_size=0.2,random_state=0)


# In[65]:


x_train.shape


# In[66]:


x_test.shape


# In[106]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[107]:


KNN=knn.fit(x_train,y_train)


# In[108]:


y_train_series = y_train.squeeze()


# In[109]:


pred_train=knn.predict(x_train)
pred_train


# In[110]:


pd.crosstab(y_train_series, pred_train, rownames = ['Actual'], colnames = ['Predictions'])


# In[111]:


print(skmet.accuracy_score(y_train_series, pred_train)) 


# In[112]:


pred = knn.predict(x_test)
pred


# In[113]:


y_test_series= y_test.squeeze()


# In[114]:


print(skmet.accuracy_score(y_test_series, pred))
pd.crosstab(y_test_series, pred, rownames = ['Actual'], colnames = ['Predictions'])


# In[116]:


cm = skmet.confusion_matrix(y_test_series, pred)
cm


# In[124]:


cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["1","2","3","5","6","7"])
cmplot.plot()
cmplot.ax_.set(title = "material calssification - confusion matrix",
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# In[126]:


acc = []
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x_train, y_train_series)
    train_acc = np.mean(neigh.predict(x_train) ==y_train_series)
    test_acc = np.mean(neigh.predict(x_test) == y_test_series)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc


# In[127]:


plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")


# In[128]:


from sklearn.model_selection import GridSearchCV
k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)


# In[129]:


grid = GridSearchCV(knn, param_grid, cv = 3, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)


# In[130]:


KNN_new = grid.fit(x_train, y_train_series)


# In[131]:


print(KNN_new.best_params_)


# In[132]:


accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )


# In[133]:


pred = KNN_new.predict(x_test)
pred


# In[134]:


cm = skmet.confusion_matrix(y_test_series, pred)


# In[135]:


cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['1', '2','3','5','6','7'])
cmplot.plot()
cmplot.ax_.set(title = 'material classification - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# In[136]:


knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))


# In[137]:


import os
os.getcwd()


# In[ ]:




