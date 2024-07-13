#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install mlxtend')
get_ipython().system('pip install feature_engine')
get_ipython().system('pip install sqlalchemy')


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sqlalchemy import create_engine
import pickle


# In[11]:


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "root", db = "books"))
user ='root'
pw = 'root'
db = 'books'


# In[12]:


data= pd.read_csv(r"C:\Program Files\datasets\book.csv", sep = ';', header = None )


# In[13]:


data.head()


# In[19]:


data.to_sql('books', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql="select * from books;"
book=pd.read_sql_query(sql,engine)


# In[20]:


book.head()


# In[21]:


import warnings
warnings.filterwarnings("ignore")


# In[23]:


book = book.iloc[:, 0].to_list()
book


# In[25]:


book_list = []
for i in book:
   book_list.append(i.split(","))
print(book_list)


# In[27]:


book_list_new = []
for i in  book_list:
   book_list_new.append(list(filter(None, i)))
print(book_list_new)


# In[29]:


TE = TransactionEncoder()
X_1hot_fit = TE.fit(book_list)


# In[30]:


pickle.dump(X_1hot_fit, open('TE.pkl', 'wb'))


# In[31]:


import os
os.getcwd()


# In[32]:


X_1hot_fit1 = pickle.load(open('TE.pkl', 'rb'))


# In[35]:


X_1hot = X_1hot_fit1.transform(book_list) 
print(X_1hot)


# In[36]:


transf_df = pd.DataFrame(X_1hot, columns = X_1hot_fit1.columns_)
transf_df
transf_df.shape


# In[38]:


count = transf_df.loc[:, :].sum()
count


# In[40]:


pop_item = count.sort_values(0, ascending = False).head(10)
pop_item


# In[41]:


pop_item = pop_item.to_frame() 


# In[42]:


pop_item = pop_item.reset_index()
pop_item


# In[43]:


pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item


# In[44]:


plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('dark_background')
pop_item.plot.barh()
plt.title('Most popular items')
plt.gca().invert_yaxis()


# In[45]:


frequent_itemsets = apriori(transf_df, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets


# In[46]:


frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets


# In[47]:


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)


# In[48]:


rules.sort_values('lift', ascending = False).head(10)


# In[52]:


def to_list(i):
    return (sorted(list(i)))
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)


# In[53]:


unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]


# In[54]:


index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
index_rules


# In[55]:


rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy


# In[56]:


rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10


# In[61]:


rules10.plot(x = "support", y = "confidence", c = rules10.lift, 
             kind = "scatter", s = 12, cmap = plt.cm.coolwarm)


# In[62]:


rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')


# In[63]:


rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")


# In[64]:


rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")


# In[65]:


rules10.to_sql('books_ar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# In[ ]:




