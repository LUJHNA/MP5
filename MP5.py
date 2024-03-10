#!/usr/bin/env python
# coding: utf-8

# # Recommender Systems

# Recommender systems provide background work
# - for users – reducing the information overload, personal assistants, social contacts
# - for business – personalizing the contact with customers, reputation systems <br>
# Modern search systems are also recommender systems

# ## Types of Systems
# __Simple recommenders__
# - offer generalized recommendations to every user, based on the popularity calculated by certain metric or score 
# 
# __Content-based recommenders__
# - makes analysis of a particular item’s features/content
# - suggests items similar to this item based on match between the user and the item<br>
# 
# __Collaborative filtering engines__
# - try to predict the rating or preference that a user would give an item based on past ratings of this user and preferences of other users<br>
# 
# __Knowledge-Based Recommendations__
# - based on related features in metadata about the product and the user

# ## 1. Simple Recommender

# Procedure: 
# - Decide on the metric to rate movies on
# - Calculate the score for every movie based on the metric
# - Sort the movies based on the score and output the top results

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.1. Data Preparation

# Pandas reads data files and loads the data into datasets. <br>The data type of the columns - dtype can be specified before the reading, or guessed by Pandas. <br>Type can be determined only after the whole file is read. As it takes lots of memory, parameter <b>low_memory=False</b> is needed.

# In[2]:


# Load data from file
data = pd.read_csv(r'C:\Users\jakob\DAT1E22Bxd\Semester4\BI BusinessIntelligence\Week1\Data\movies_metadata.csv', sep=',', low_memory=False)
# data = pd.read_csv('http://www.kaggle.com/rounakbanik/the-movie-dataset', low_memory=False)


# In[3]:


# Check the size of data
data.shape


# In[4]:


# Print the first three rows to get idea about the data
data.head(3)


# In[5]:


# See which movies are rated
data['title']


# In[6]:


# see the columns and column types
data.info()


# In[7]:


# Check the statistics of the numeric data
data.describe()


# In[8]:


# See the distribution of votes
data['vote_count'].hist(bins=80)


# The histogram shows that most movies have few ratings. <br>Movies with most ratings are those that are most famous.

# In[9]:


# See the distribution of the ratings
data['vote_average'].hist(bins=30)


# The histogram shows that most of the movies are rated between 5 and 8

# In[10]:


# Check the relationship between the rating of a movie and the number of ratings. 
# We do this by plotting a scatter plot using seaborn
import seaborn as sb
sb.jointplot(x=data['vote_average'], y=data['vote_count'], data=data)


# In[11]:


data.sort_values('vote_average', ascending=False).head(10)


# In[12]:


# Exclude all rows, where there is no votes or the number of votes is below a specified minimum
m = data['vote_count'].quantile(0.90)
print(m)


# In[13]:


# Make a copy and filter out the qualified movies into a new DataFrame
q_movies = data.copy().loc[data['vote_count'] >= m]
q_movies.shape


# In[14]:


data['vote_average']


# In[15]:


# Get the average value of all ratings in column 'vote_average'
C = data['vote_average'].mean()
print(C)


# Compute the <b>weighted average rating</b> of each movie as a new feature
# 
#         WAR = v/(v+m)*R + m/(v+m)*C
# 
# where<br>
# 
#     R is the average Rating of this movie - votes_average<br>
#     C is the average vote across the whole report, Currently <br>
#     v is the number of votes for this movie - votes_count<br>
#     m is the minimum number of votes required for a movie to be listed in the chart<br>    
#     
# This is the IMDb formula for calculating the Top Rated 250 titles,  
# https://www.imdb.com/chart/top?ref_=nb_mv_3_chttp

# In[16]:


def war(x, m=m, c=C):
    v = x['vote_count']
    r = x['vote_average']
    return (v/(v+m) * r) + (m/(m+v) * c)


# In[17]:


# Define the new feature 'score'
# Calculate its values for the qualified dataframe with war()
q_movies['score'] = q_movies.apply(war, axis=1)
print(q_movies['score'])


# In[18]:


#Sort movies in the dataframe based on the scores calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)


# ## 2. Content-Based Recommender 

# #### System that recommends movies similar to a particular movie
# The idea is that 
# - similar users share the same interest and 
# - similar items are liked by one user
# 
# <br>Two types:
# - User-based: measure the similarity between target users and other users
# - Item-based: measure the similarity between the target items and other items

# In[19]:


# Description of a movie stays in the field/feature called 'overview'
# Print overviews of the first 5 movies to see the format
data['overview'].head()


# Doc format above is unappropriate for comparisson, needs transformation
# 
# We create word vector <b>Term Frequency-Inverse Document Frequency</b> (TF-IDF) for each overview.<br>
# TF-IDF score shows the frequency of a word occurring in a document, 
# down-weighted by the number of documents in which it occurs
# 
# For all documents, we create a matrix, where 
# 
#     each column represents one word in the overview vocabulary (all the words that appear in at least one document)
#     each row represents one movie

# In[20]:


# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer


# ### 2.1. Prepare the data

# In[21]:


# Define a TF-IDF Vectorizer Object, while removing all english stop words such as 'the', 'a', ...
tfidf = TfidfVectorizer(stop_words='english')

# Replace all NaN with an empty string
data['overview'] = data['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['overview'])

# It takes some time


# In[22]:


# Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[23]:


# Compute a similarity score
# We can implement either euclidean, Pearson, or cosine similarity scores

# To use cosine score, import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix for each vs each movie
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[24]:


print(cosine_sim)


# In[25]:


# We need reverse mapping of movies and indices: to identify movie by index
# Construct a reverse map of indices and movie titles
indices = pd.Series(data.index, index=data['title']).drop_duplicates()


# In[26]:


print(indices)


# ### 2.2. Define a Function

# Define a function that takes in a movie __title__ as an input and outputs a list of the 10 most similar movies
# by identifying the __index__ of a movie in your metadata DataFrame, given its title
# 1. Get the index of the movie given its title.
# 2. Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list of tuples where the first element is its __position__ and the second is the similarity score.
# 3. Sort the aforementioned list of tuples based on the __similarity scores__; that is, the second element.
# 4. Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar        to a particular movie is the movie itself).
# 5. Return the titles corresponding to the indices of the top elements.

# In[27]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


# In[44]:


get_ipython().system('pip install joblib')


# In[46]:


import joblib

# Save the TF-IDF vectorizer
joblib.dump(tfidf, r'C:\Users\jakob\DAT1E22Bxd\Semester4\BI BusinessIntelligence\Week1\MP5\tfidf_vectorizer.joblib')

# Save the cosine similarity matrix
joblib.dump(cosine_sim, r'C:\Users\jakob\DAT1E22Bxd\Semester4\BI BusinessIntelligence\Week1\MP5\cosine_similarity_matrix.joblib')

# Save the indices mapping
joblib.dump(indices, r'C:\Users\jakob\DAT1E22Bxd\Semester4\BI BusinessIntelligence\Week1\MP5\indices_mapping.joblib')


# ### 2.3. Test the Recommender

# In[28]:


get_recommendations('The Shawshank Redemption')


# In[29]:


get_recommendations('Life Is Beautiful')


# In[30]:


get_recommendations('Star Trek')


# In[36]:


get_ipython().system('pip install streamlit')

