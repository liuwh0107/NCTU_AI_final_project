#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Comment 
# Preprocessing-> Vectorize:CountVectorizer, Stemming(works,worked,working-> work): snowballstemmer, 
#              -> tokenize(word_tokenize), stopwords
# Model-> Gradient boost
# Function-> preprocessing_stemming_cleaning,clean_text (they do preprocessing above)
# Public score-> 0.78271
# Reference-> https://towardsdatascience.com/detecting-disaster-from-tweets-classical-ml-and-lstm-approach-4566871af5f7


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from sklearn import model_selection, metrics, preprocessing, ensemble, model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# In[3]:


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(each_text):

    # remove URL from text
    each_text_no_url = re.sub(r"http\S+", "", each_text)
    
    # remove numbers from text
    text_no_num = re.sub(r'\d+', '', each_text_no_url)

    # tokenize each text
    word_tokens = word_tokenize(text_no_num)
    
    # remove sptial character
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))

    # remove stop words and lower
    text_with_no_stop_word = [w.lower() for w in clean_text if not w in stop_words]  

    # do stemming
    stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]
    
    return " ".join(" ".join(stemmed_text).split())


# In[4]:


#train is a bool to indicate whether it's train or test turn
def preprocessing_stemming_cleaning(raw_data,train):
    if train:
        raw_data['word_count'] = raw_data['text'].apply(lambda x: len(x.split(" ")) )
        raw_data = raw_data[raw_data['word_count']>2]
        raw_data = raw_data.reset_index()
    
    raw_data['clean_text'] = raw_data['text'].apply(lambda x: clean_text(x) )
    raw_data['keyword'] = raw_data['keyword'].fillna("none")
    raw_data['clean_keyword'] = raw_data['keyword'].apply(lambda x: clean_text(x) )
    raw_data['keyword_text'] = raw_data['clean_keyword'] + " " + raw_data["clean_text"]
    
#     feature = 'keyword_text'
#     label = "target"
    return raw_data


# In[5]:


# Rreading train dataset
file_path = "./train.csv"
raw_data = pd.read_csv(file_path)
# print("Data points count: ", raw_data['id'].count())
# raw_data.head()


# In[6]:


# Plotting target value counts
plt.figure(figsize=(10,8))
ax = raw_data['target'].value_counts().sort_values().plot(kind="bar")
ax.grid(axis="y")
plt.suptitle("Target Value Counts", fontsize=20)
plt.show()


# In[7]:


print("Number of missing data for column keyword: ", raw_data['keyword'].isna().sum())
print("Number of missing data for column location: ", raw_data['location'].isna().sum())
print("Number of missing data for column text: ", raw_data['text'].isna().sum())
print("Number of missing data for column target: ", raw_data['target'].isna().sum())


# In[8]:


plt.figure(figsize=(15,8))
sns.heatmap(raw_data.drop('id', axis=1).isnull(), cbar=False, cmap="GnBu").set_title("Missing data for each column")
plt.show()


# In[9]:


#plt.figure(figsize=(15,8))
#raw_data['word_count'] = raw_data['text'].apply(lambda x: len(x.split(" ")) )
# sns.distplot(raw_data['word_count'].values, hist=True, kde=True, kde_kws={"shade": True})
# plt.axvline(raw_data['word_count'].describe()['25%'], ls="--")
# plt.axvline(raw_data['word_count'].describe()['50%'], ls="--")
# plt.axvline(raw_data['word_count'].describe()['75%'], ls="--")

# plt.grid()
# plt.suptitle("Word count histogram")
# plt.show()

# remove rows with under 3 words
# raw_data = raw_data[raw_data['word_count']>2]
# raw_data = raw_data.reset_index()


# In[10]:


# print("25th percentile: ", raw_data['word_count'].describe()['25%'])
# print("mean: ", raw_data['word_count'].describe()['50%'])
# print("75th percentile: ", raw_data['word_count'].describe()['75%'])


# In[11]:


# Clean text columns
# stop_words = set(stopwords.words('english'))
# stemmer = SnowballStemmer('english')


# def clean_text(each_text):

#     # remove URL from text
#     each_text_no_url = re.sub(r"http\S+", "", each_text)
    
#     # remove numbers from text
#     text_no_num = re.sub(r'\d+', '', each_text_no_url)

#     # tokenize each text
#     word_tokens = word_tokenize(text_no_num)
    
#     # remove sptial character
#     clean_text = []
#     for word in word_tokens:
#         clean_text.append("".join([e for e in word if e.isalnum()]))

#     # remove stop words and lower
#     text_with_no_stop_word = [w.lower() for w in clean_text if not w in stop_words]  

#     # do stemming
#     stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]
    
#     return " ".join(" ".join(stemmed_text).split())


# raw_data['clean_text'] = raw_data['text'].apply(lambda x: clean_text(x) )
# raw_data['keyword'] = raw_data['keyword'].fillna("none")
# raw_data['clean_keyword'] = raw_data['keyword'].apply(lambda x: clean_text(x) )


# In[12]:


print(stop_words)
print(stemmer)


# In[13]:


# Combine column 'clean_keyword' and 'clean_text' into one
#raw_data['keyword_text'] = raw_data['clean_keyword'] + " " + raw_data["clean_text"]


# In[14]:


feature = 'keyword_text'
label = "target"

#get raw_data
raw_data = preprocessing_stemming_cleaning(raw_data,True)
# split train and test
X_train, X_test,y_train, y_test = model_selection.train_test_split(raw_data[feature],
                                                                   raw_data[label],
                                                                   test_size=0.3,
                                                                   random_state=0, 
                                                                   shuffle=True)


# In[15]:


print(raw_data.keys())
X_test.head()
print(type(X_test))


# In[16]:


X_train_GBC = X_train.values.reshape(-1)
x_test_GBC = X_test.values.reshape(-1)
print(x_test_GBC)


# In[17]:


# Vectorize text:: COuntVectorizer()
vectorizer = CountVectorizer()
X_train_GBC = vectorizer.fit_transform(X_train_GBC)
x_test_GBC = vectorizer.transform(x_test_GBC)


# In[18]:


# Train the model
model = ensemble.GradientBoostingClassifier(learning_rate=0.1,                                            
                                            n_estimators=2000,
                                            max_depth=9,
                                            min_samples_split=6,
                                            min_samples_leaf=2,
                                            max_features=8,
                                            subsample=0.9)
model.fit(X_train_GBC, y_train)


# In[19]:


# Evaluate the model
predicted_prob = model.predict_proba(x_test_GBC)[:,1]
predicted = model.predict(x_test_GBC)

accuracy = metrics.accuracy_score(predicted, y_test)
print("Test accuracy: ", accuracy)
print(metrics.classification_report(y_test, predicted, target_names=["0", "1"]))
print("Test F-scoare: ", metrics.f1_score(y_test, predicted))


# In[ ]:





# In[20]:


# Plot confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, predicted)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, cbar=False, cmap='Reds', annot=True, fmt='d')
ax.set(xlabel="Predicted Value", ylabel="True Value", title="Confusion Matrix")
ax.set_yticklabels(labels=['0', '1'], rotation=0)

plt.show()


# In[32]:


## TEST ##
path = "./test.csv"
data = pd.read_csv(path)
data.head()


# In[33]:


feature = 'keyword_text'
label = "target"
#get test data after preprocessing
data =  preprocessing_stemming_cleaning(data,False)


# In[34]:



test = data[feature].values.reshape(-1)
test = vectorizer.transform(test)


# In[35]:


predicted = model.predict(test)


# In[36]:


print(predicted)


# In[37]:


sub = pd.DataFrame()
sub["id"] = data['id']
sub["target"] = predicted


# In[38]:


sub


# In[39]:


sub.to_csv("subgbs.csv", index = False)


# In[ ]:





# In[ ]:




