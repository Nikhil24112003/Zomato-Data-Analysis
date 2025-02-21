#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import sqlite3


# In[4]:


con = sqlite3.connect(r"C:\Users\chaud\OneDrive\Desktop\Projects\Zomato_Data_Analysis_project\zomato_rawdata.sqlite")


# In[5]:


df = pd.read_sql_query("SELECT * FROM Users" , con)


# In[6]:


df.head(2)


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


## dealing with missing values ....


# In[10]:


df.head(2)


# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[13]:


df.isnull().sum()/len(df)*100


# In[14]:


df['rate'].unique()


# In[15]:


df['rate'].replace(('NEW' , '-') , np.nan , inplace=True)


# In[16]:


df['rate'].unique()


# In[17]:


"4.1/5".split('/')[0]


# In[18]:


type("4.1/5".split('/')[0])


# In[19]:


float("4.1/5".split('/')[0])


# In[20]:


df['rate'] = df['rate'].apply(lambda x : float(x.split('/')[0]) if type(x)==str else x)


# In[21]:


df['rate']


# In[22]:


## Analysing a relation between online order option and rating of the restaurant 



# In[23]:


x = pd.crosstab(df['rate'] , df['online_order'])


# In[24]:


x


# In[25]:


x.plot(kind='bar' , stacked=True) 


# In[26]:


x


# In[27]:


x.sum(axis=1).astype(float)


# In[28]:


normalize_df = x.div(x.sum(axis=1).astype(float) , axis=0)


# In[29]:


(normalize_df*100).plot(kind='bar' , stacked=True)


# In[30]:


## Data Cleaning to perform Text Analysis


# In[31]:


df['rest_type'].isnull().sum()


# In[32]:


data = df.dropna(subset=['rest_type'])


# In[33]:


data['rest_type'].isnull().sum()


# In[34]:


data['rest_type'].unique()


# In[35]:


quick_bites_df = data[data['rest_type'].str.contains('Quick Bites')]


# In[36]:


quick_bites_df.shape


# In[37]:


quick_bites_df.columns


# In[38]:


#### a) Perform Lower-case operation 


# In[39]:


quick_bites_df['reviews_list']


# In[40]:


quick_bites_df['reviews_list'] = quick_bites_df['reviews_list'].apply(lambda x:x.lower())


# In[41]:


from nltk.corpus import RegexpTokenizer


# In[42]:


tokenizer = RegexpTokenizer("[a-zA-Z]+")


# In[43]:


tokenizer


# In[44]:


tokenizer.tokenize(quick_bites_df['reviews_list'][3])


# In[45]:


sample = data[0:10000]


# In[46]:


reviews_tokens = sample['reviews_list'].apply(tokenizer.tokenize)


# In[47]:


## Performing Unigram analysis & removal of stopwords


# In[48]:


#### c) Removal of stopwords from data


# In[49]:


reviews_tokens


# In[50]:


from nltk.corpus import stopwords


# In[51]:


stop = stopwords.words('english')


# In[52]:


print(stop)


# In[53]:


stop.extend(['rated' , "n" , "nan" , "x" , "RATED" , "Rated"])


# In[54]:


print(stop)


# In[55]:


reviews_tokens


# In[56]:


rev3 = reviews_tokens[3]
print(rev3)


# In[57]:


print([token for token in rev3 if token not in stop])


# In[58]:


reviews_tokens_clean = reviews_tokens.apply(lambda each_review : [token for token in each_review if token not in stop])


# In[59]:


reviews_tokens_clean


# In[60]:


type(reviews_tokens_clean)


# In[61]:


total_reviews_2D = list(reviews_tokens_clean)


# In[62]:


total_reviews_1D = []

for review in total_reviews_2D:
    for word in review:
        total_reviews_1D.append(word)


# In[63]:


total_reviews_1D


# In[64]:


from nltk import FreqDist


# In[65]:


fd = FreqDist()


# In[66]:


for word in total_reviews_1D:
    fd[word] = fd[word] + 1


# In[67]:


fd.most_common(20)


# In[68]:


fd.plot(20)


# In[69]:


## Performing Bi-gram & Trigram analysis on data


# In[70]:


from nltk import FreqDist , bigrams , trigrams


# In[71]:


bi_grams = bigrams(total_reviews_1D)


# In[72]:


bi_grams


# In[73]:


fd_bigrams = FreqDist()

for bigram in bi_grams:
    fd_bigrams[bigram] = fd_bigrams[bigram] + 1


# In[74]:


fd_bigrams.most_common(20)


# In[75]:


fd_bigrams.plot(20)


# In[76]:


fd_bigrams.most_common(100)


# In[77]:


## Trigram Analysis


# In[78]:


tri_grams = trigrams(total_reviews_1D)


# In[79]:


fd_trigrams = FreqDist()

for trigram in tri_grams:
    fd_trigrams[trigram] = fd_trigrams[trigram] + 1


# In[80]:


fd_trigrams.most_common(50)


# In[81]:


###  Extract geographical-coordinates from data


# In[82]:


df.head(3)


# In[83]:


get_ipython().system('pip install geocoder')
get_ipython().system('pip install geopy')


# In[84]:


df['location']


# In[85]:


df['location'].unique()


# In[86]:


len(df['location'].unique())


# In[87]:


df['location'] = df['location'] + " , Bangalore  , Karnataka , India "


# In[88]:


df['location']


# In[89]:


df['location'].unique()


# In[90]:


df_copy = df.copy()


# In[91]:


df_copy['location'].isnull().sum()


# In[92]:


df_copy = df_copy.dropna(subset=['location'])


# In[93]:


df_copy['location'].isnull().sum()


# In[94]:


locations = pd.DataFrame(df_copy['location'].unique())


# In[95]:


locations.columns = ['name']


# In[96]:


locations


# In[97]:


from geopy.geocoders import Nominatim


# In[98]:


geolocator = Nominatim(user_agent="app" , timeout=None)


# In[99]:


lat=[]
lon=[]

for location in locations['name']:
    location = geolocator.geocode(location)
    if location is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(location.latitude)
        lon.append(location.longitude)


# In[100]:


locations['latitude'] = lat
locations['longitude'] = lon


# In[101]:


locations


# In[102]:


###   build geographical Heat-Maps


# In[103]:


locations.isnull().sum()


# In[104]:


locations[locations['latitude'].isna()]


# In[105]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[106]:


locations['latitude'][45] = 12.9764122
locations['longitude'][45] = 77.6017437


# In[107]:


locations[locations['latitude'].isna()]


# In[108]:


locations['latitude'][79] = 13.0163
locations['longitude'][79] = 77.6785


# In[109]:


locations['latitude'][85] = 13.0068
locations['longitude'][85] = 77.5813


# In[110]:


df['cuisines'].isnull().sum()


# In[111]:


df = df.dropna(subset=['cuisines'])


# In[112]:


north_india = df[df['cuisines'].str.contains('North Indian')]


# In[113]:


north_india.shape


# In[114]:


north_india.head(2)


# In[115]:


north_india_rest_count = north_india['location'].value_counts().reset_index().rename(columns={'index':'name' , "location":"count"})


# In[116]:


north_india_rest_count


# In[117]:


locations


# In[118]:


heatmap_df = north_india_rest_count.merge(locations , on='name' , how='left')


# In[119]:


heatmap_df


# In[120]:


import folium


# In[121]:


basemap = folium.Map()


# In[122]:


basemap


# In[123]:


heatmap_df.columns


# In[124]:


from folium.plugins import HeatMap


# In[125]:


HeatMap(heatmap_df[['latitude', 'longitude' , "count"]]).add_to(basemap)


# In[126]:


basemap


# In[127]:


def get_heatmap(cuisine):
    cuisine_df = df[df['cuisines'].str.contains(cuisine)]
    
    cuisine_rest_count = cuisine_df['location'].value_counts().reset_index().rename(columns={'index':'name' , "location":"count"})
    heatmap_df = cuisine_rest_count.merge(locations , on='name' , how='left')
    print(heatmap_df.head(4))
    
    basemap = folium.Map()
    HeatMap(heatmap_df[['latitude', 'longitude' , "count"]]).add_to(basemap)
    return basemap


# In[128]:


get_heatmap('South Indian')


# In[129]:


df['cuisines'].unique()


# In[ ]:




