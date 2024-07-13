#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# In[ ]:


game=pd.read_csv(r"C:\Program Files\datasets\game.csv",encoding='utf8')


# In[ ]:


from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "root", db = "game"))
game.to_sql('game', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# In[ ]:


sql = 'select * from game'
game = pd.read_sql_query(sql, con = engine)


# In[ ]:


game.isnull().sum()


# In[ ]:


game['game']=game['game'].fillna("unknown")
game['rating']=game['rating'].fillna(3.5)


# In[ ]:


tfidf = TfidfVectorizer(stop_words = "english") 
tfidf


# In[ ]:


tfidf_matrix = tfidf.fit(game.game)  
tfidf_matrix


# In[ ]:


joblib.dump(tfidf_matrix, 'matrix')


# In[ ]:


os.getcwd()


# In[ ]:


mat = joblib.load("matrix")
tfidf_matrix = mat.transform(game.game)
tfidf_matrix.shape 


# In[ ]:


cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
game_index = pd.Series(game.index, index = game['game']).drop_duplicates()


# In[ ]:


game_id=game_index['Grand Theft Auto IV']
game_id


# In[ ]:


topN = 5
def get_recommendations(Name, topN):
    game_id = game_index[game]
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN + 1]
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    game_similar_show = pd.DataFrame(columns = ["game", "rating"])
    game_similar_show["game"] = game.loc[anime_idx, "game"]
    game_similar_show["rating"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    return(game_similar_show.iloc[1:, ])


# In[ ]:


rec = get_recommendations("Grand Theft Auto IV", topN = 10)
rec


# In[ ]:




