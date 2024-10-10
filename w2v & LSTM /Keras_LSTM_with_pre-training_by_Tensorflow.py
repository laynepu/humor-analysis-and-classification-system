# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:10:28 2018

@author: LaynelMoon
"""

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
#nltk.download('punkt')
from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))
def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index
                

#corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
## convert to lower case
#corpus_raw = corpus_raw.lower()
#
#words = []
#for word in corpus_raw.split():
#    if word != '.': # because we don't want to treat . as a word
#        words.append(word)
#words = set(words) # so that all duplicate words are removed
#word2int = {}
#int2word = {}
#vocab_size = len(words) # gives the total number of unique words
#for i,word in enumerate(words):
#    word2int[word] = i
#    int2word[i] = word


# raw sentences is a list of sentences.
#raw_sentences = corpus_raw.split('.')
#sentences = []
#for sentence in raw_sentences:
#    sentences.append(sentence.split())
#print('sentences',sentences)
    

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./alphabet_new1.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = sentence.split()
#        print(words)
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
#print('word_freqs',word_freqs)
#print('max_len ',maxlen)
#print('nb_words ', len(word_freqs))
#print('......',word_freqs['......'])
## 準備數據
MAX_FEATURES = 1000
MAX_SENTENCE_LENGTH = 40
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2int = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
#print(word2int)
#print('word_index',word_index)
#print('word_freqs.most_common(MAX_FEATURES)',word_freqs.most_common(MAX_FEATURES))
word2int["PAD"] = 0
word2int["UNK"] = 1
index2word = {v:k for k, v in word2int.items()}
#print('index2word',index2word)
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
# 讀取訓練資料，將每一單字以 dictionary 儲存
with open('./alphabet_new1.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = sentence.split()
        seqs = []
        for word in words:
            if word in word2int:
                seqs.append(word2int[word])
            else:
                seqs.append(word2int["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
#print(X)
#print(y)



words = []
sentences = []
with open("alphabet_new1.txt","r",encoding="utf8") as fp:
    for sentence in fp:
#        print(sentence)
        sentence=sentence[2:len(sentence)]
        for word in sentence.split():
            if word in word2int:
                words.append(word)
        sentences.append(sentence.split())
words = set(words)
#print(words)
#word2int = {}
#int2word = {}
##vocab_size = len(words) # gives the total number of unique words
#for i,word in enumerate(words):
#    word2int[word] = i+2
#    int2word[i] = word

data = []
WINDOW_SIZE = 5
for sentence in sentences:
    for word_index, word in enumerate(sentence):
#        print(sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1])
        if word in word2int:
            for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
                if nb_word != word and nb_word in word2int:
                    data.append([word, nb_word])
#                print(word,nb_word)
#print(data)
# function to convert numbers to one hot vectors

x_train = [] # input word
y_train = [] # output word
for data_word in data:
#    print(word2int[data_word[1]])
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
#print(y_train)
# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
#print(x)
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_test = tf.Variable(tf.random_normal([5,3]))
x_test = tf.Variable(tf.random_normal([2,2776]))
test = tf.reduce_mean(x_test)

EMBEDDING_DIM = 10 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM],seed=1)) #bias
#print(W1)
hidden_representation = tf.add(tf.matmul(x,W1), b1)
#print(tf.matmul(x,W1),11)
#print(b1)
#print(hidden_representation)
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
#print(prediction)
#init=tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
##    print(sess.run(b1))
##    print(sess.run(b1))
##    print(sess.run(b1))
#    print(sess.run(tf.add(W1,b1)))
##    print(sess.run(W1.assign(tf.matmul(x,W1),feed_dict={x:x_train})))
#    
#    print(sess.run(b1))
#    print(sess.run(prediction,feed_dict={x:x_train}))
#
#
##    print(sess.run(y_test))
#    print(sess.run(x_test))
#    print(sess.run(test))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 1
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
                
vectors = sess.run(W1 + b1)
#print(vectors)
#
#
#print(int2word[find_closest(word2int['一天'], vectors)])
#print(int2word[find_closest(word2int['老師'], vectors)])
#print(int2word[find_closest(word2int['跌倒'], vectors)])                
#

              


# 字句長度不足補空白        
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH) #大於maxlen會被截短，反之在後面補0
# 資料劃分訓練組及測試組
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# 模型構建
EMBEDDING_SIZE = EMBEDDING_DIM
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
model = Sequential()
# 加『嵌入』層
model.add(Embedding(vocab_size, EMBEDDING_SIZE,weights=[vectors],input_length=MAX_SENTENCE_LENGTH))
# 加『LSTM』層
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# binary_crossentropy:二分法
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

# 模型訓練
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

# 預測
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('預測','真實','句子'))
for i in range(10):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    
# 模型存檔
model.save('keras_LSTM.h5')  # creates a HDF5 file 'model.h5'
    
#### 自己輸入測試
INPUT_SENTENCES = ['秝瑋 要 去 睡覺 了'
                   ,'我餓了，想要去吃飯了'
                   ,'老師 誰 的 優點 最 多 甲 大華 老師 為 什麼 甲 因為 他家 開 藥局 '
                   , '父親 剛 開學 考試 你 怎么 就 得 個 ‘ 0 ’ 分 ? 兒子 老師 說 我們 一切 都 要 從 ‘ 0 ’ 開始'
                   ]
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
# 轉換文字為數值
i=0
for sentence in  INPUT_SENTENCES:
    words_temp = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words_temp:
        if word in word2int:
            seq.append(word2int[word])
        else:
            seq.append(word2int['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# 預測，並將結果四捨五入，轉換為 0 或 1
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {1:'是笑話-', 0:'不是笑話-'}
# 顯示結果
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))  
                



model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)
normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')

fig, ax = plt.subplots()
for word in words:
#    print(word,vectors[word2int[word]][0] ,vectors[word2int[word]][1])
#    try:
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
#        print(word)
#    except:
#        continue
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.show()
                

