#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


get_ipython().system('pip install imblearn')


# In[3]:


get_ipython().system('conda install scikit-learn=0.24.2')
get_ipython().system('pip install -U imbalanced-learn')



# In[4]:


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skmet
import joblib


# In[5]:


data=pd.read_csv("C:\Program Files\datasets\SalaryData_Train.csv")


# In[6]:


from sqlalchemy import create_engine
user='root'
pw='root'
db='salary'
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[7]:


data.to_sql('salary_train',con= engine,if_exists='replace',chunksize=1000,index=False)


# In[8]:


sql='select * from salary_train'
train= pd.read_sql_query(sql,engine)


# In[9]:


test_data=pd.read_csv("C:\Program Files\datasets\SalaryData_Test.csv")


# In[10]:


test_data.to_sql('salary_test',con=engine,if_exists='replace',chunksize=1000,index=False)


# In[11]:


test_sql='select * from salary_test'
test=pd.read_sql_query(test_sql,engine)


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


train.head()


# In[15]:


train.Salary.value_counts()
train.Salary.value_counts()/len(train.Salary)


# In[16]:


categorical = [var for var in train.columns if train[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)


# In[17]:


train[categorical].head()


# In[18]:


train[categorical].isnull().sum()


# In[19]:


for var in categorical: 
    print(train[var].value_counts())


# In[20]:


train.workclass.unique()


# In[21]:


train.workclass.value_counts()


# In[22]:


train.occupation.unique()


# In[23]:


train.occupation.value_counts()


# In[24]:


train.native.unique()


# In[25]:


train.native.value_counts()


# In[26]:


for var in categorical:
    print(var, ' contains ', len(train[var].unique()), ' labels')


# In[27]:


numerical = [var for var in train.columns if train[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[28]:


train[numerical].head()


# In[29]:


train[numerical].isnull().sum()


# In[30]:


train[numerical].isnull().sum()


# In[31]:


X = train.drop(['Salary'], axis=1)
y = train['Salary']


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[33]:


X_train.shape, X_test.shape


# In[34]:


X_train.dtypes


# In[35]:


X_test.dtypes


# In[36]:


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical


# In[37]:


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical


# In[38]:


X_train[categorical].isnull().mean()


# In[39]:


for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))


# In[40]:


for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)  


# In[41]:


X_train[categorical].isnull().sum()


# In[42]:


X_test[categorical].isnull().sum()


# In[43]:


X_train.isnull().sum()


# In[44]:


get_ipython().system('pip install category_encoders')
import category_encoders as ce


# In[45]:


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])
X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[47]:


X_train.shape


# In[48]:


X_test.head()


# In[49]:


X_test.shape


# In[50]:


cols = X_train.columns


# In[51]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[52]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[53]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[54]:


X_train.head()


# In[55]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[56]:


y_pred = gnb.predict(X_test)
y_pred


# In[57]:


from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[58]:


y_pred_train = gnb.predict(X_train)
y_pred_train


# In[59]:


print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# In[61]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


# In[64]:


import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[65]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[66]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[67]:


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[ ]:




