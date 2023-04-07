#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information
# 
# The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.
# 
# Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.
# 
# ## Import Modules

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# ## Loading the Dataset

# In[37]:


df = pd.read_csv('train_E6oV3lV.csv')
df


# In[38]:


df.head()


# In[39]:


df.info()


# # Preprocessing Dataset

# In[40]:


# remove patterns in the input text

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt


# In[41]:


# removing twitter handles(@user)
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'],"@[\w]*")


# In[42]:


df.head()


# In[43]:


# remove special characters, numbers and punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
df.head()


# In[44]:


# remove short words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
df.head()


# In[45]:


# individual words considered as tokens
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet


# In[46]:


# stem the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_tweet.head()


# In[47]:


# convert into a single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
df['clean_tweet'] = tokenized_tweet
df.head()


# # Exploratory Data Analysis

# In[48]:


get_ipython().system('pip install wordcloud')


# In[52]:


# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height = 500, random_state =42, max_font_size =100).generate(all_words)

#plot the graph
plt.figure(figsize = (13,6))
plt.imshow(wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[53]:


# frequent word visualization for +ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==0]])

wordcloud = WordCloud(width = 800, height = 500, random_state =42, max_font_size =100).generate(all_words)

#plot the graph
plt.figure(figsize = (13,6))
plt.imshow(wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[54]:


# frequent word visualization for -ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])

wordcloud = WordCloud(width = 800, height = 500, random_state =42, max_font_size =100).generate(all_words)

#plot the graph
plt.figure(figsize = (13,6))
plt.imshow(wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[57]:


#extract the hashtags
def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


# In[58]:


# extract hashtags from non racist/sexist tweets
ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])

# extract hashtags from racist/sexist tweets
ht_negative = hashtag_extract(df['clean_tweet'][df['label']==1])


# In[60]:


ht_positive[:5]


# In[61]:


ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])


# In[62]:


ht_positive[:5]


# In[63]:


freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                   'Count' : list(freq.values())})
d.head()


# In[64]:


# select top 10 hashtags

d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data =d, x = 'Hashtag', y ='Count')
plt.show()


# In[65]:


freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                   'Count' : list(freq.values())})
d.head()


# In[66]:


# select top 10 hashtags

d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data =d, x = 'Hashtag', y ='Count')
plt.show()


# # Input Split

# In[80]:


# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.90, min_df =2, max_features = 1000, stop_words = 'english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])


# In[82]:


bow[0].toarray()


# In[83]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state =42,test_size=0.25)


# # Model Training

# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# In[85]:


#training
model = LogisticRegression()
model.fit(x_train, y_train)


# In[86]:


#testing
pred = model.predict(x_test)
f1_score(y_test, pred)


# In[87]:


accuracy_score(y_test, pred)


# In[89]:


# use probability to get output
pred_prob = model.predict_proba(x_test)
pred = pred_prob[:, 1] >= 0.3
pred = pred.astype(np.int32)
f1_score(y_test, pred)


# In[90]:


accuracy_score(y_test, pred)


# In[92]:


pred_prob[0][1] >= 0.3

