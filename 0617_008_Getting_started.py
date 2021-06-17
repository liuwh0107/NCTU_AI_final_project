#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Comment
# Preprocessing-> Use count vectorizer
# Model-> ridge classifier [linear model]
#Function -> count_vectorizer
# Public score-> 0.78026
# Reference->https://www.kaggle.com/philculliton/nlp-getting-started-tutorial


# In[22]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# In[23]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[26]:


def count_vectorizer(train_df):
    count_vectorizer = feature_extraction.text.CountVectorizer()

    ## let's get counts for the first 5 tweets in the data
    example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
    train_vectors = count_vectorizer.fit_transform(train_df["text"])

    ## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
    # that the tokens in the train vectors are the only ones mapped to the test vectors - 
    # i.e. that the train and test vectors use the same set of tokens.
    test_vectors = count_vectorizer.transform(test_df["text"])
    return train_vectors, test_vectors


# In[30]:


train_vectors, test_vectors = count_vectorizer(train_df)


# In[31]:


## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()


# In[32]:


scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores


# In[33]:


clf.fit(train_vectors, train_df["target"])


# In[34]:


sample_submission = pd.read_csv("sample_submission.csv")


# In[35]:


sample_submission["target"] = clf.predict(test_vectors)


# In[36]:


sample_submission.head()


# In[37]:


sample_submission.to_csv("submission_q.csv", index=False)


# In[ ]:





# In[ ]:




