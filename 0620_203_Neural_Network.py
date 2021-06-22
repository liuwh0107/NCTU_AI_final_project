# Comment 
# field -> text, keyword
# preprocess
# Model-> LSTM(Siamese Network)
# Reference ->  https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E4%B8%80%E5%80%8B%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%EF%BC%8C%E5%85%A9%E5%80%8B%E6%96%B0%E8%81%9E%E6%A8%99%E9%A1%8C


from functools import total_ordering
from os import remove
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
#import keras
import string
import heapq
import nltk
import re
import contractions
from emoticon_fix import emoticon_fix
import warnings
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


class PredictionCallback(keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    #x, y = self.test_data
    print('epoch ',epoch+1)
    global x_train, y_train, x_val, x2_val, y_val, x1_test, x2_test, ans, validation_acc, test_acc
    loss1, acc1 = self.model.evaluate([x_val, x2_val], y_val, verbose=0)
    print('\nvalidation loss: {}, acc: {}\n'.format(loss1, acc1))
    validation_acc.append(acc1)
    y_pred = self.model.predict([x1_test,x2_test])
    pred = np.argmax(y_pred, axis=1)
    #print('prediction: {} at epoch: {}'.format(y_pred, epoch))
    #print(classification_report(list(pred),list(ans['target'])))
    acc2 = accuracy_score(list(ans['target']),list(pred))
    test_acc.append(acc2)
    print('test accuracy: ',acc2)

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma



def preprocess(str1):
    res = nltk.tokenize.word_tokenize(str1)
    str2 = ''
    for token in res:
        if token not in nltk_stopwords:
            lemm = lemmatize(token)
            try:
                lower_str = lemm.lower()
            except:
                lower_str = lemm
            str2 += lower_str + ' '
    return str(str2)
    
warnings.filterwarnings("ignore")
MAX_NUM_WORDS = 8842
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 
                                   lower=True, split=' ', char_level=False,num_words=MAX_NUM_WORDS)
tokenizer2 = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 
                                   lower=True, split=' ', char_level=False,num_words=MAX_NUM_WORDS)
nltk_stopwords = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.WordNetLemmatizer()

train_df = pd.read_csv("./dataset/train.csv")
test_df = pd.read_csv("./dataset/test.csv")
ans = pd.read_csv("./answer.csv")


# train_df['clean_text'] = train_df['text'].apply(preprocess2)
train_df["clean_text1"]=train_df['text'].apply(lambda x :emoticon_fix.emoticon_fix(x))
train_df["clean_text2"]=train_df['clean_text1'].apply(lambda x :contractions.fix(x))
train_df["clean_text3"]=train_df['clean_text2'].apply(lambda x :preprocess(x))

train_df['keyword'] = train_df['keyword'].replace(np.nan, '', regex=True)
train_df["clean_keyword1"]=train_df['keyword'].apply(lambda x :emoticon_fix.emoticon_fix(x))
train_df["clean_keyword2"]=train_df['clean_keyword1'].apply(lambda x :contractions.fix(x))
train_df["clean_keyword3"]=train_df['clean_keyword2'].apply(lambda x :preprocess(x))

# test_df['clean_text'] = test_df['text'].apply(preprocess2)
test_df["clean_text1"]=test_df['text'].apply(lambda x :emoticon_fix.emoticon_fix(x))
test_df["clean_text2"]=test_df['clean_text1'].apply(lambda x :contractions.fix(x))
test_df["clean_text3"]=test_df['clean_text2'].apply(lambda x :preprocess(x))

test_df['keyword'] = test_df['keyword'].replace(np.nan, '', regex=True)
test_df["clean_keyword1"]=test_df['keyword'].apply(lambda x :emoticon_fix.emoticon_fix(x))
test_df["clean_keyword2"]=test_df['clean_keyword1'].apply(lambda x :contractions.fix(x))
test_df["clean_keyword3"]=test_df['clean_keyword2'].apply(lambda x :preprocess(x))


train1 = list(train_df["clean_text3"])
test1 = list(test_df["clean_text3"])
train2 = list(train_df["clean_keyword3"])
test2 = list(test_df["clean_keyword3"])
tmp_list = train1 + test1 + train2 + test2
tmp_df = pd.DataFrame()
tmp_df['text'] = pd.Series(tmp_list)
# tmp_list2 = train2 + test2
# tmp_df2 = pd.DataFrame()
# tmp_df['keyword'] = pd.Series(tmp_list)

# tmp = tokenizer.fit_on_texts(train_df['clean_text3'])
# tmp2 = tokenizer2.fit_on_texts(test_df['clean_text3'])
tmp = tokenizer.fit_on_texts(tmp_df['text'])
# tmp2 = tokenizer2.fit_on_texts(test_df['keyword'])

#print(tmp)
x1_train = tokenizer.texts_to_sequences(train_df['clean_text3'])
x2_train = tokenizer.texts_to_sequences(train_df['clean_keyword3'])
x1_test = tokenizer.texts_to_sequences(test_df['clean_text3'])
x2_test = tokenizer.texts_to_sequences(test_df['clean_keyword3'])

len1 = max([len(seq) for seq in x1_train])
len2 = max([len(seq) for seq in x1_test])
max_seq_len = max(len1,len2)

len3 = max([len(seq) for seq in x2_train])
len4 = max([len(seq) for seq in x2_test])
max_seq_len2 = max(len3,len4)

MAX_SEQUENCE_LENGTH = max_seq_len
x1_train = keras.preprocessing.sequence.pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
x2_train = keras.preprocessing.sequence.pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)

x1_test = keras.preprocessing.sequence.pad_sequences(x1_test, maxlen=MAX_SEQUENCE_LENGTH)
x2_test = keras.preprocessing.sequence.pad_sequences(x2_test, maxlen=MAX_SEQUENCE_LENGTH)
# print(x1_train[:1])
y1_train = keras.utils.to_categorical(train_df['target'])


VALIDATION_RATIO = 0.2

x_train, x_val, x2_train, x2_val, y_train, y_val = model_selection.train_test_split( x1_train,x2_train, y1_train, test_size=VALIDATION_RATIO, random_state=1)



dic = {}
idx = 0
pic_epoch =[]
NUM_EPOCHS = 5
v = []
t = []
for i in range(0,NUM_EPOCHS):
    pic_epoch.append(i+1)

for dim in [128,256]:
    for batch in [8,16,32]:
        # print(idx,dim,batch)
        dic[idx] = [dim,batch]

        NUM_EMBEDDING_DIM = dim # 128, 256, 1024
        #MAX_NUM_WORDS = len(tmp2['word_counts'].split(',')) + 1 # <23060
        NUM_CLASSES = 2
        NUM_LSTM_UNITS = 128
        embedding_layer = keras.layers.Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
        top_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
        top_embedded = embedding_layer(top_input)
        bm_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
        bm_embedded = embedding_layer(bm_input)

        lstm = keras.layers.LSTM(NUM_LSTM_UNITS)
        output = lstm(top_embedded)
        output2 = lstm(bm_embedded)
        merged = keras.layers.concatenate([output, output2], axis=-1)
        # softmax -> normalize -> the value is between 0 and 1 , sum = 1
        dense = keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
        predictions = dense(merged)
        model = keras.models.Model(inputs = [top_input,bm_input], outputs = predictions)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        BATCH_SIZE = batch
        validation_acc = []
        test_acc = []
        history = model.fit( x=[x_train, x2_train], y=y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=([x_val,x2_val], y_val),callbacks=[PredictionCallback()], shuffle=True)
        v.append(validation_acc)
        t.append(test_acc)
        print(len(v),v[idx])
        print(len(t),t[idx])
        idx += 1

h, w = 30, 30        # for raster image
nrows, ncols = 2, 3  # array of sub-plots
figsize = [8, 8]     # figure size, inches

# prep (x,y) for extra plotting on selected sub-plots
xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
ys = np.abs(np.sin(xs))           # absolute of sine

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
for i,axs in enumerate(ax.flat):
    axs.plot(pic_epoch, v[i], color = 'blue', marker='o', linestyle = '--', label='Validation')
    axs.plot(pic_epoch, t[i], color = 'orange', marker='o', linestyle = '-', label='Test')
    axs.set_xlabel('Epoch', color = 'red')
    axs.set_ylabel('Accuracy', color = 'red')
    dim, batch = dic[i]
    title = 'Dim = ' + str(dim) + ', ' + 'Batch = '+ str(batch)
    axs.set_title(title)
plt.tight_layout()
plt.show()            


#font1 = {'family':'serif','color':'blue','size':20}
# plt.plot(pic_epoch, validation_acc, color = 'blue', marker='o', linestyle = '--', label='Validation')
# plt.plot(pic_epoch, test_acc, color = 'orange', marker='o', linestyle = '-', label='Test')
# plt.legend(loc = 'upper left')
# plt.xlabel('Epoch', color = 'red')
# plt.ylabel('Accuracy', color = 'red')
# plt.title('Accuracy per epoch', color = 'red')
# plt.show()