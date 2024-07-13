#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


from sqlalchemy import create_engine


# In[3]:


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="root",
                               db="flight"))


# In[4]:


connecting_route=pd.read_csv(r"C:\Program Files\datasets\connecting_routes1.csv")


# In[5]:


connecting_route.to_sql('connecting_route',con= engine,if_exists='replace',chunksize=1000,index=False)


# In[6]:


sql='select * from connecting_route'
connecting_route = pd.read_sql_query(sql,con=engine)


# In[7]:


connecting_route.head()


# In[8]:


connecting_route=connecting_route.iloc[0:100,0:8]
connecting_route.columns


# In[9]:


for_g = nx.graph
for_g=nx.from_pandas_edgelist(connecting_route,source='main Airport',target='Destination')


# In[10]:


print("number of nodes:",nx.number_of_nodes(for_g))
print("number of edges:",nx.number_of_edges(for_g))


# In[11]:


connected_components = list(nx.connected_components(for_g))
for i, component in enumerate(connected_components, start=1):
    subgraph = for_g.subgraph(component)
    
    data = pd.DataFrame({
        "closeness": pd.Series(nx.closeness_centrality(subgraph)),
        "Degree": pd.Series(nx.degree_centrality(subgraph)),
        "eigenvector": pd.Series(nx.eigenvector_centrality(subgraph)),
        "betweenness": pd.Series(nx.betweenness_centrality(subgraph))
    })

    print(f"Connected Component {i}:")
    print(data)


# In[12]:


subgrpah= nx.Graph()
subgraph= nx.from_pandas_edgelist(connecting_route, source = 'main Airport', target = 'Destination')


# In[13]:


f = plt.figure()
pos = nx.spring_layout(for_g, k = 0.015)
nx.draw_networkx(for_g, pos, ax=f.add_subplot(111), node_size = 15, node_color = 'red')
plt.show()


# In[14]:


f.savefig("graph.png")


# In[15]:


flight_hault=pd.read_csv(r"C:\Program Files\datasets\flight_hault.csv")


# In[16]:


flight_hault.to_sql('flight_hault',con= engine,if_exists='replace',chunksize=1000,index=False)


# In[17]:


sql='select * from flight_hault'
flight_hault = pd.read_sql_query(sql,con=engine)


# In[18]:


flight_hault.isnull().sum()


# In[19]:


mode=flight_hault['IATA_FAA'].mode


# In[50]:


flight_hault['IATA_FAA'].fillna('mode')


# In[52]:


flight_hault=flight_hault.iloc[0:100,0:12]
flight_hault.columns


# In[53]:


for_g_hault= nx.graph
for_g_hault=nx.from_pandas_edgelist(flight_hault,source='City',target='City')


# In[54]:


print("number of nodes:",nx.number_of_nodes(for_g_hault))
print("number of edges:",nx.number_of_edges(for_g_hault))


# In[55]:


nx.is_connected(for_g_hault)


# In[56]:


connected_components_hault = list(nx.connected_components(for_g_hault))
for i, component in enumerate(connected_components_hault, start=1):
    subgraph_hault = for_g_hault.subgraph(component)
    
    data_hault = pd.DataFrame({
        "closeness": pd.Series(nx.closeness_centrality(subgraph_hault)),
        "Degree": pd.Series(nx.degree_centrality(subgraph_hault)),
        "eigenvector": pd.Series(nx.eigenvector_centrality(subgraph_hault)),
        "betweenness": pd.Series(nx.betweenness_centrality(subgraph_hault))
    })

    print(f"Connected Component {i}:")
    print(data_hault)


# In[57]:


subgrpah_hault= nx.Graph()
subgraph_hault= nx.from_pandas_edgelist(flight_hault, source = 'City', target = 'City')


# In[58]:


f_hault = plt.figure()
pos = nx.spring_layout(for_g, k = 0.015)
nx.draw_networkx(for_g, pos, ax=f.add_subplot(111), node_size = 15, node_color = 'red')
plt.show()


# In[ ]:





# In[ ]:




