#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install feature_engine')
get_ipython().system('pip install sklearn_pandas')


# In[2]:


import pandas as pd 
import sweetviz
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import joblib
import pickle


# In[3]:


from sqlalchemy import create_engine 
kmeans=pd.read_csv(r"C:\Program Files\datasets\AutoInsurance (2).csv")
user="root"
pw="root"
db="kmean"
engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# In[4]:


kmeans.to_sql("insurance",con=engine,if_exists="replace",chunksize=1000,index= False)
sql="select * from insurance;"
df=pd.read_sql_query(sql,engine)


# In[5]:


df.info()


# In[6]:


report = sweetviz.analyze([df, "df"])
report.show_html('Report.html')


# In[7]:


df1=["Customer Lifetime Value","Income","Education","Monthly Premium Auto","Vehicle Class","Vehicle Size","Sales Channel","Coverage",
    "EmploymentStatus","Policy","Policy Type","Total Claim Amount"]


# In[8]:


data=df[df1]
data.head()


# In[9]:


data["Customer Lifetime Value"].value_counts


# In[10]:


data.Income.value_counts


# In[11]:


data.Education.value_counts


# In[12]:


data["Monthly Premium Auto"].value_counts


# In[14]:


data["Vehicle Class"].value_counts


# In[15]:


data["Vehicle Size"].value_counts


# In[16]:


data["Sales Channel"].value_counts


# In[17]:


data.Coverage.value_counts


# In[18]:


data["EmploymentStatus"].value_counts


# In[19]:


data.isnull().sum()


# In[20]:


import seaborn as sns
sns.pairplot(data,hue="Sales Channel")


# In[21]:


data=df[df1]


# In[22]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data["Education"]=label.fit_transform(data["Education"])
data["Vehicle Class"]=label.fit_transform(data["Vehicle Class"])
data["Vehicle Size"]=label.fit_transform(data["Vehicle Size"])
data["Sales Channel"]=label.fit_transform(data["Sales Channel"])
data["Coverage"]=label.fit_transform(data["Coverage"])
data["Policy"]=label.fit_transform(data["Policy"])
data["Policy Type"]=label.fit_transform(data["Policy Type"])
data["EmploymentStatus"]=label.fit_transform(data["EmploymentStatus"])


# In[23]:


categorical_data=data.select_dtypes(include=["object"]).columns
categorical_data


# In[46]:


from sklearn.preprocessing import MinMaxScaler
numerical_pipeline=Pipeline([('impute',SimpleImputer(strategy='mean')),('scale',MinMaxScaler())])
numerical_pipeline


# In[47]:


numerical_pipeline.fit_transform(data)


# In[48]:


numerical_data=data.select_dtypes(exclude=['object']).columns


# In[49]:


data.head()


# In[50]:


joblib.dump(data,'data')


# In[51]:


import os
os.getcwd()


# In[52]:


cleaned=pd.DataFrame(data,columns=data.columns)


# In[53]:


print(cleaned.iloc[:,:])


# In[54]:


from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                 tail="both",
                 fold=1.5,
                 variables=["Customer Lifetime Value","Income","Education","Monthly Premium Auto","Vehicle Class",
                           "Sales Channel","Coverage","EmploymentStatus",
                            "Policy","Total Claim Amount"])


# In[55]:


outliers=winsor.fit(cleaned[["Customer Lifetime Value","Income","Education","Monthly Premium Auto","Vehicle Class",
                           "Sales Channel","Coverage","EmploymentStatus",
                            "Policy","Total Claim Amount"]])


# In[56]:


cleaned[["Customer Lifetime Value","Income","Education","Monthly Premium Auto","Vehicle Class",
                           "Sales Channel","Coverage","EmploymentStatus",
                            "Policy","Total Claim Amount"]]=outliers.transform(cleaned[["Customer Lifetime Value","Income","Education","Monthly Premium Auto",
                           "Vehicle Class","Sales Channel","Coverage","EmploymentStatus",
                            "Policy","Total Claim Amount"]])


# In[57]:


cleaned.head()


# In[58]:


cleaned.plot(kind="box",subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=2)
plt.show()


# In[59]:


TWSS = []
clusters = list(range(2, 10))
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(cleaned)
    TWSS.append(kmeans.inertia_)
TWSS


# In[60]:


plt.plot(clusters, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")


# In[61]:


model = KMeans(n_clusters =2)
yy = model.fit(cleaned)


# In[62]:


model.labels_


# In[65]:


metrics.silhouette_score(cleaned, model.labels_)


# In[63]:


silhouette_coefficients = []
for k in range (2,10):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(cleaned)
    score = metrics.silhouette_score(cleaned, model.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])


# In[64]:


sorted(silhouette_coefficients, reverse = True, key = lambda x: x[1])


# In[66]:


bestmodel = KMeans(n_clusters = 2)
result = bestmodel.fit(cleaned)


# In[67]:


pickle.dump(result, open('Clust_kmean.pkl', 'wb'))


# In[68]:


import os
os.getcwd()


# In[69]:


bestmodel.labels_


# In[70]:


mb = pd.Series(bestmodel.labels_)


# In[82]:


df_clust = pd.concat([mb,df, data], axis = 1)
df_clust= cleaned.rename(columns = {0:'cluster_id'})
df_clust.head()


# In[85]:


cluster_agg = cleaned.iloc[:,3:].groupby(mb).mean()
cluster_agg


# In[87]:


cleaned.to_csv('AutoInsurance (2).csv', encoding = 'utf-8', index = False)
 


# In[88]:


import os
os.getcwd()


# In[53]:


cleaned.plot(kind="box",subplots=True,sharey=False,figsize=(10,6))


# In[57]:


duplicate=cleaned.duplicated()
print(duplicate)


# In[58]:


sum(duplicate)


# In[59]:


print(cleaned.shape)


# In[61]:


cleaned_df=cleaned.drop_duplicates()
print(cleaned_df)


# In[63]:


correlation_matrix=cleaned_df.corr()
correlation_matrix


# In[65]:


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
ac = AgglomerativeClustering(2, linkage = 'average')
ac_clusters = ac.fit_predict(cleaned_df)


# In[66]:


km = KMeans(2)
km_clusters = km.fit_predict(cleaned_df)


# In[223]:


db_param_options = [
    [1950, 240], [1960, 245], [1970, 250], [1980, 255], [1990, 260],
    [2000, 265], [2010, 270], [2020, 275], [2030, 280], [2040, 285],
    [2050, 290], [2060, 295], [2070, 300], [2080, 305], [2090, 310],
    [2100, 315], [2110, 320], [2120, 325], [2130, 330], [2140, 335],
    [2150, 340], [2160, 345], [2170, 350]
]


# In[227]:


from sklearn.metrics import silhouette_score
import numpy as np
for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(cleaned_df)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(cleaned_df, db_clusters))


# In[228]:


db = DBSCAN(eps =2120, min_samples = 325)
db_clusters = db.fit_predict(cleaned_df,db_clusters)


# In[229]:


plt.figure(1)
plt.title("insurance Clusters from Agglomerative Clustering")
plt.scatter(cleaned_df['Income'], cleaned_df['Policy Type'], c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()


# In[230]:


plt.figure(2)
plt.title("insurance Clusters from K-Means")
plt.scatter(cleaned_df['Income'], cleaned_df['Customer Lifetime Value'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()


# In[231]:


plt.figure(3)
plt.title("insurance Clusters from K-Means")
plt.scatter(cleaned_df['Income'], cleaned_df['Monthly Premium Auto'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()


# In[232]:


plt.figure(4)
plt.title("insurance Clusters from K-Means")
plt.scatter(cleaned_df['Income'], cleaned_df['Total Claim Amount'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()


# In[233]:


print("Agg Clustering: ", silhouette_score(cleaned_df,ac_clusters))



# In[234]:


print("K-Means Clustering: ", silhouette_score(cleaned_df, km_clusters))


# In[235]:


print("DBSCAN Clustering: ", silhouette_score(cleaned_df, db_clusters))


# In[236]:


import pickle
pickle.dump(db, open('db.pkl', 'wb'))


# In[237]:


model = pickle.load(open('db.pkl', 'rb'))


# In[238]:


res = model.fit_predict(cleaned_df)


# In[ ]:




