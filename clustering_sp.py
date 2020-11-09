#!/usr/bin/env python
# coding: utf-8

# **Clustering Playlist Database from Spotify API**

# In[2]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_playlist_original = pd.read_csv('C:/Users/Espadas/IronHack-OCT2020-DA/3rd week/3rd day/dataframe_spotify_playlist1000.csv')
df_playlist = pd.read_csv('C:/Users/Espadas/IronHack-OCT2020-DA/3rd week/3rd day/dataframe_spotify_playlist1000.csv')
df_playlist_original.drop(['0', 'track_href'], axis = 1, inplace=True)
df_playlist_original.dropna(inplace=True)


# In[4]:


df_playlist.drop(['0', 'analysis_url', 'id','track_href','type','uri'], axis = 1, inplace=True)
df_playlist.dropna(inplace=True)


# In[5]:


#initilize transformer
scaler = StandardScaler()


# In[6]:


#fit & transform the data
X_scaled = scaler.fit_transform(df_playlist)


# pd.DataFrame(X_scaled).head()

# **Clustering**

# In[7]:


#Inizilite the  model
kmeans = KMeans(n_clusters=8, random_state=1234)


# In[8]:


#fit the model
kmeans.fit(X_scaled)


# In[9]:


# Predicting / assigning the clusters:
clusters = kmeans.predict(X_scaled)


# In[10]:


# Check the size of the clusters
pd.Series(clusters).value_counts()


# In[11]:


df_playlist['cluster'] = clusters
df_playlist['cluster'].head()


# In[12]:


df_playlist.groupby('cluster').mean()


# #### Tuning KMeans

# In[13]:


for n in [1, 5, 100]:
    for i in [2, 300]:
        for alg in ["full", "elkan"]:
            kmeans = KMeans(n_clusters=8,
                    init="random", 
                    n_init=n,  # try with 1, 4, 8, 20, 30, 100...,
                    max_iter=i,
                    tol=0,
                    algorithm=alg,
                    random_state=1234)
            kmeans.fit(X_scaled)
            print("n_init:", n, "| max_iter:", i, "| alg:", alg, "| inertia:", kmeans.inertia_)


# ***Finding the optimal number of clusters***

# In[14]:


# Try to run Kmeans with all values of K, from 2 to 20
K = range(2, 15)

# For each model, store the inertia in a list
inertia = []

for k in K:
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# In[15]:


# Plot the results
plt.figure(figsize=(16,8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Elbow Method showing the optimal k')


# In[16]:


# repeat the same process, now with the silhouette score
K = range(2, 25)
silhouette = []

for k in K:
    clusterer = KMeans(n_clusters=k)
    preds= clusterer.fit_predict(X_scaled)
    centers = clusterer.cluster_centers_
    score= silhouette_score(X_scaled, preds)
    silhouette.append(score)
print(silhouette)


# In[17]:


# plot silhouette_score
plt.figure(figsize=(16,8))
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Elbow Method showing the optimal k')


# In[18]:


kmeans = KMeans(n_clusters=11)


# In[19]:


kmeans.fit(X_scaled)


# In[20]:


clustered_ft = kmeans.predict(X_scaled)
clustered_ft


# In[21]:


cluster_col = pd.DataFrame(pd.Series(clustered_ft))
cluster_col


# In[ ]:





# In[22]:


df_playlist['cluster'] = cluster_col


# In[23]:


df_playlist[['id','uri']] = df_playlist_original[['id','uri']]


# In[24]:


df_playlist


# In[ ]:




