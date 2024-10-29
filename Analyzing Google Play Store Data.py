#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[33]:


df = pd.read_csv('F:\\mahmoud ali\\oasis project\\Task5\\apps.csv')


# In[34]:


print(df.head())


# In[35]:


print(df.info())


# In[36]:


print(df.isnull().sum())


# In[37]:


df['Size'] = pd.to_numeric(df['Size'], errors='coerce')


# In[38]:


df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')


# In[39]:


df.dropna(inplace=True)


# In[40]:


category_counts = df['Category'].value_counts()


# In[41]:


plt.figure(figsize=(12, 8))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Apps by Category')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.show()


# In[42]:


print(df[['Rating', 'Size', 'Installs', 'Price']].describe())


# In[43]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[44]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Size'], bins=30, kde=True)
plt.title('Distribution of App Sizes')
plt.xlabel('Size (MB)')
plt.ylabel('Frequency')
plt.show()


# In[45]:


sid = SentimentIntensityAnalyzer()


# In[46]:


df['sentiment'] = df['Reviews'].apply(lambda x: sid.polarity_scores(str(x))['compound'])


# In[47]:


df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))


# In[48]:


sentiment_counts = df['sentiment_label'].value_counts()


# In[49]:


plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution of App Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[50]:


get_ipython().system('pip install plotly')


# In[51]:


import plotly.express as px


# In[52]:


fig = px.pie(df, names='Category', title='App Distribution by Category')
fig.show()


# Recommendtions
# Focus on Popular Categories: Investing in app development within popular categories can increase chances of success.
# Enhance App Ratings: Work on improving apps with lower ratings to enhance user experience.
# Address Negative Feedback: Prioritize addressing negative feedback to resolve technical issues or enhance features.

# In[ ]:




