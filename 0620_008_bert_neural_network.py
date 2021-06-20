#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.kaggle.com/arnab132/disaster-tweets-prediction-using-nlp
# https://www.tensorflow.org/official_models/fine_tuning_bert
# https://huggingface.co/transformers/model_doc/bert.html
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
#plt.style.use('fivethirtyeight')

import re
import string

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore')

#Import Dataset

data = pd.read_csv('train.csv')

data.head()

values = data['target'].value_counts().values
# fig = go.Figure(data=[go.Pie(labels=['Count 0','Count 1',], values=values)])
# fig.update_layout(template="plotly_dark",title={'text': "Count of Type",'y':0.9,
#                                                 'x':0.45,'xanchor': 'center','yanchor': 'top'},
#                   font=dict(size=18, color='white', family="Courier New, monospace"))
# fig.show()

def limpa_texto(data):
    tx = data.apply(lambda x: re.sub("http\S+", '', str(x)))
    tx = tx.apply(lambda x: re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '',x))
    tx = tx.apply(lambda x: re.sub(' +', ' ', x)) # remover espaços em brancos
    tx = tx.apply(lambda x: re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', x)) # remover as hashtag
    tx = tx.apply(lambda x: re.sub('(@[A-Za-z]+[A-za-z0-9-_]+)', '', x)) # remover os @usuario
    tx = tx.apply(lambda x: re.sub('rt', '', x)) # remover os rt
    tx = tx.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    return tx
data['text'] = limpa_texto(data['text'])
data.head()


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

def bert_encode(data, maximum_length) :
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)

data['message_len'] = data['text'].apply(lambda x: len(x.split(' ')))
data.head()

#fig = px.histogram(data, x='message_len')
#fig.update_layout(template="plotly_dark",title={'text': "Phrase Length",'y':0.9,
#                                                'x':0.45,'xanchor': 'center','yanchor': 'top'},
#                  font=dict(size=18, color='white', family="Courier New, monospace"))
#fig.show()

texts = data['text']
target = data['target']

train_input_ids, train_attention_masks = bert_encode(texts,30)
from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-base-uncased')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

#Neural Network Model

def create_model(bert_model):
    
    input_ids = tf.keras.Input(shape=(30,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(30,),dtype='int32')

    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(32,activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model(bert_model)
                     
model.summary()

stoped = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)
                     
redutor = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
                     
history = model.fit([train_input_ids, train_attention_masks],
    target, validation_split=0.2, epochs=2, batch_size=16, callbacks=[stoped, redutor])

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend(['Train and Test Accuracy Plot'])
axes[0].grid(True)

axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Error')
axes[1].legend(['Train and Test Loss Plot'])
axes[1].grid(True)


# In[12]:



model.summary()


# In[13]:


print(history.history.keys())


# In[14]:


test_data =  pd.read_csv('test.csv')


# In[15]:


test_data['text'] = limpa_texto(test_data['text'])
test_data['message_len'] = test_data['text'].apply(lambda x: len(x.split(' ')))


# In[16]:


test_texts = test_data['text']
test_input_ids, test_attention_masks = bert_encode(test_texts,30)
predicted = model.predict([test_input_ids,test_attention_masks])
print(predicted)


# In[7]:


for i in predicted:
    print(i)


# In[8]:


result=list()
for element in predicted:
    if element[0]>0.5:
        result.append(1)
    else:
        result.append(0)


# In[9]:



print(result)


# In[10]:


sub = pd.DataFrame()
sub["id"] = test_data['id']
sub["target"] = result
sub


# In[17]:


sub.to_csv("submission_bert_pre2.csv", index=False)


# In[ ]:




