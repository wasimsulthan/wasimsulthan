#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.metrics as skmet
import joblib, pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from urllib.parse import quote 


# In[ ]:


loan=pd.read_csv("C:\\Program Files\\datasets\\loan.csv")
user='root'
pw='root'
db='loan'
engine=(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
loan.to_sql("loan",con = engine,chunksize=1000,if_exists='replace',index=False)
sql="select * from loan"
df=pd.read_sql_query(sql,engine)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df=loan.dropna(axis=1)
df


# In[ ]:


df.loan_status.value_counts()


# In[ ]:


df=loan.drop(['member_id', 'id','term','grade','sub_grade','int_rate',
                'application_type' ,'issue_d', 'pymnt_plan','url','zip_code',
                'addr_state', 'earliest_cr_line', 'initial_list_status','initial_list_status', 'recoveries', 'collection_recovery_fee'], axis=1)


# In[ ]:


df.head()


# In[ ]:


import sweetviz as sv
report = sv.analyze(df)
report.show_html("report.html")


# In[ ]:


correalition=df.corr()
correalition


# In[ ]:


df.columns


# In[ ]:


X = loan.drop(['loan_status','policy_code','acc_now_delinq','delinq_amnt'], axis = 1)
Y = loan['loan_status']


# In[ ]:


numerical_features= X.select_dtypes(exclude=['object']).columns
numerical_features


# In[ ]:


categorical_features=X.select_dtypes(include=["object"]).columns
categorical_features


# In[ ]:


pipeline1=Pipeline([("impute",SimpleImputer(strategy='mean')),("scale",MinMaxScaler())])
pipeline1


# In[ ]:


pipeline2=Pipeline([('onehot',OneHotEncoder(sparse_output=True))])
pipeline2


# In[ ]:


processed=ColumnTransformer([("numerical",pipeline1,numerical_features),
                            ("categorical",pipeline2,categorical_features)])
processed


# In[ ]:


imp_enc_scale = processed.fit(X)


# In[ ]:


joblib.dump(imp_enc_scale, 'imp_enc_scale')


# In[ ]:


cleandata = pd.DataFrame(imp_enc_scale.transform(X), columns = imp_enc_scale.get_feature_names_out())
cleandata.columns


# In[ ]:


lowvariance = ['num__delinq_2yrs', 'num__pub_rec', 'num__out_prncp', 'num__out_prncp_inv', 'num__total_rec_late_fee']


# In[ ]:


continous = ['num__loan_amnt', 'num__funded_amnt', 'num__funded_amnt_inv',
       'num__installment', 'num__annual_inc', 'num__dti', 
       'num__inq_last_6mths', 'num__open_acc', 
       'num__revol_bal', 'num__total_acc', 
        'num__total_pymnt', 'num__total_pymnt_inv',
       'num__total_rec_prncp', 'num__total_rec_int', 
       'num__last_pymnt_amnt']


# In[ ]:


cleandata.iloc[:, 0:22].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# In[ ]:


winsor = Winsorizer(capping_method = 'iqr', 
                          tail = 'both', 
                          fold = 1.5,
                          variables = continous)


# In[ ]:


winsor2 = Winsorizer(capping_method = 'gaussian', 
                          tail = 'both',
                          fold = 0.05,
                          variables = lowvariance)


# In[ ]:


outlier = winsor.fit(cleandata[continous])


# In[ ]:


joblib.dump(outlier, 'winsor')


# In[ ]:


cleandata[continous] = outlier.transform(cleandata[continous])


# In[ ]:


outlier2 = winsor2.fit(cleandata[lowvariance])


# In[ ]:


joblib.dump(outlier2, 'winsor2')


# In[ ]:


cleandata[lowvariance] = outlier2.transform(cleandata[lowvariance])


# In[ ]:


cleandata.iloc[:, 0:22].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
plt.subplots_adjust(wspace = 0.75)
plt.show()


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(cleandata, Y, test_size = 0.2, random_state = 0)


# In[ ]:


model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train_X, train_Y)


# In[ ]:


test_predict = model.predict(test_X) 


# In[ ]:


accuracy_score(test_Y, test_predict)


# In[ ]:


train_predict = model.predict(train_X)


# In[ ]:


accuracy_score(train_Y, train_predict) 


# In[ ]:


logmodel1 = LogisticRegression(multi_class = "multinomial")
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'None'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


# In[ ]:


clf = GridSearchCV(logmodel1, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(train_X, train_Y)


# In[ ]:


best_clf.best_estimator_


# In[ ]:


print (f'Accuracy - : {best_clf.score(train_X, train_Y):.3f}')
print (f'Accuracy - : {best_clf.score(test_X,test_Y):.3f}')


# In[ ]:


best_clf1 = clf.fit(cleandata, Y)


# In[ ]:


best_model = best_clf1.best_estimator_


# In[ ]:


print (f'Accuracy - : {best_clf1.score(cleandata, Y):.3f}')
print (f'Accuracy - : {best_clf1.score(test_X,test_Y):.3f}')


# In[ ]:


pickle.dump(best_model,open('multinomial.pkl','wb'))


# In[ ]:




