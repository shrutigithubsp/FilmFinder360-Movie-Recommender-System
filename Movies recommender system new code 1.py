#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib
import os


# In[47]:


movies = pd.read_csv('Dataset.csv')


# In[48]:


movies.head


# In[49]:


movies.shape


# In[50]:


movies.info()


# In[51]:


movies.describe()


# In[52]:


movies.isnull().sum()


# In[53]:


movies.dropna(subset=['ID'],inplace=True)


# In[54]:


movies.isnull().sum()


# In[55]:


movies=movies[['ID','original_title','Genre','Overview']]


# In[56]:


movies.head()


# In[57]:


movies.duplicated().sum()


# In[58]:


movies.drop_duplicates(inplace=True)


# In[59]:


movies.duplicated().sum()


# In[60]:


movies['tags'] =movies['Overview']+movies['Genre']


# In[61]:


movies.head


# In[62]:


movies.describe()


# In[63]:


movies.info()


# In[64]:


new_data  = movies.drop(columns=['Overview', 'Genre',])
new_data


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer


# In[66]:


cv=CountVectorizer(max_features=10000, stop_words='english')
cv


# In[67]:


CountVectorizer(max_features=10000, stop_words='english')
vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
vector.shape


# In[68]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity[0].shape


# In[69]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[70]:


new_data.head(2)


# In[71]:


def recommend(movie):
    movie_index = new_data[new_data['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_data.iloc[i[0]].original_title)


# In[72]:


recommend('avatar')


# In[73]:


recommend('free guy')


# In[74]:


recommend('happy new year')


# In[75]:


recommend('chennai express')


# In[76]:


recommend('bodyguard')


# In[77]:


recommend('tubelight')


# In[78]:


recommend('prey')


# In[79]:


recommend('cocktail')


# In[80]:


type(new_data)


# In[81]:


import pickle
import pandas as pd

# Load the data from the pickle file
with open('movies_list.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Check if the loaded data is a DataFrame
if isinstance(loaded_data, pd.DataFrame):
    # Convert the DataFrame to a string
    data_as_string = loaded_data.to_csv(index=False, encoding='utf-8')

    # Save the data as a UTF-8 encoded text file
    with open('movies_list.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(data_as_string)
else:
    print("Loaded data is not a DataFrame.")


# In[82]:


import pickle
import pandas as pd

# Load the data from the pickle file
with open('similarity.pkl', 'rb') as file:
    similarity_data = pickle.load(file)

# Check if the loaded data is a DataFrame
if isinstance(similarity_data, pd.DataFrame):
    # Convert the DataFrame to a string
    data_as_string = similarity_data.to_string(index=False)  # Set index=False to exclude row indices

    # Save the data as a UTF-8 encoded text file
    with open('output_similarity.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(data_as_string)
else:
    print("Loaded data is not a DataFrame.")


# In[83]:


recommend('zindagi na milegi dobara')


# In[84]:


recommend('iron man 2')


# In[85]:


recommend('venom')


# In[86]:


recommend('f9')


# In[87]:


recommend('toy story')


# In[88]:


recommend('toy story 2')


# In[89]:


recommend('sex tape')


# In[90]:


recommend('titanic')


# In[48]:


recommend('harry potter and the goblet of fire')


# In[ ]:





# In[ ]:




