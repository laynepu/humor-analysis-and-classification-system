# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 04:38:04 2019

@author: LaynelMoon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:55:58 2018

@author: LaynelMoon
"""
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#import nltk
#nltk.download('punkt')
#from sklearn import preprocessing
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from gensim.models import word2vec
import time
import numpy as np
seed = 7
np.random.seed(seed)


#maxlen = 0
#word_freqs = collections.Counter()
#num_recs = 0
#with open('./alphabe t_new.txt','r+', encoding='UTF-8') as f:
#    for line in f:
#        label, sentence = line.strip().split("\t")
#        words = sentence.split()
##        print(words)
#        if len(words) > maxlen:
#            maxlen = len(words)
#        for word in words:
#            word_freqs[word] += 1
#        num_recs += 1
#        

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.LineSentence("alphabet_new1.txt")
sentences=[]
num_recs = 0
with open('./學習電子報+zol笑話大全_data_for_model_training.txt','r+', encoding='UTF-8') as f:
    for line in f:
        sentence = line[2:len(line)].strip().split()
        sentences.append(sentence)
        num_recs += 1
#print(sentences)
EMBEDDING_SIZE = 10
model_w2v = word2vec.Word2Vec(sentences, size=EMBEDDING_SIZE, window=5)
word2int={}
word2int["PAD"] = 0
word2int["UNK"] = 1
vector=[]
temp_vector=[]
for i in range(EMBEDDING_SIZE):
    temp_vector.append(0)
vector.append(temp_vector)
vector.append(temp_vector)
vocab_size=2
for index,word in enumerate(model_w2v.wv.vocab):
    word2int[word]=index+2
    vector.append(model_w2v.wv[word])
    vocab_size+=1
#print('index=',index)
index2word = {v:k for k, v in word2int.items()}
vectors=np.array(vector)

X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./學習電子報+zol笑話大全_data_for_model_training.txt','r+', encoding='UTF-8') as f:
    for line in f:
#        print(line)
        label, sentence = line.strip().split("\t")
        words_temp = sentence.split()
#        print(words)
#        time.sleep(10)
        seqs = []
        for word in words_temp:
            if word in word2int:
                seqs.append(word2int[word])
            else:
#                print(word)
                seqs.append(word2int["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
#print(len(word2int))
time.sleep(10)                   
                    
#保存模型，供日後使用
#model_w2v.save("gensim_word2vec_學習電子報+zol笑話大全_data_for_model_training.model")
#模型讀取方式
# model = word2vec.Word2Vec.load("your_model_name")
    
MAX_SENTENCE_LENGTH = 40

# 字句長度不足補空白        
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH) #大於maxlen會被截短，反之在後面補0

# 資料劃分訓練組及測試組
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# 模型構建
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
i=0
for train, test in kfold.split(X, y):
    print("fold{}",format(i))
    model = Sequential()
    
    # 加『嵌入』層
    model.add(Embedding(vocab_size, EMBEDDING_SIZE,weights=[vectors],input_length=MAX_SENTENCE_LENGTH))
    
    # 加『LSTM』層
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    #precision = as_keras_metric(tf.metrics.precision)
    #recall = as_keras_metric(tf.metrics.recall)
    
    # binary_crossentropy:二分法
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    
    model.fit(X[train], y[train], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X[test], y[test]))



# 模型訓練
#model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

#model = load_model('keras_LSTM_gensim_sg_equal.h5')

    # 預測
    #score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    score, =accuracy = model.evaluate(X[test], y[test], batch_size=BATCH_SIZE)

    print("accuracy: %.2f%%" % (accuracy*100))
    cvscores.append(accuracy*100)
    i+=1
print("acuracy_mean=%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# 模型存檔
#model.save('keras_LSTM_gensim_sg_學習電子報+zol笑話大全_data_for_model_training.h5')  # creates a HDF5 file 'model.h5'


print("\nTest score: %.3f, accuracy:%.3f" % (score, accuracy))
#print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('預測','真實','句子'))
for idx in range(len(Xtest)):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
#    if int(round(ypred)) != int(ylabel):
#        print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    

    
#### 自己輸入測試
#INPUT_SENTENCES = [ '用於 說服 反對 建造 新 主力艦 的 閣員 '
#                   ,'不過 由於 整個 銀河系 的 戰況 吃緊'
#                   ,'老婆 說 嫁給 魔鬼 也 比 嫁給 你好 老公 說 不行 啊 近親 不能 結婚 啊'
#                   ,'秝瑋 要 去 睡覺 了'
#                   ,'我餓了，想要去吃飯了'
#                   ,'老師 誰 的 優點 最 多 甲 大華 老師 為 什麼 甲 因為 他家 開 藥局 '
#                   , '父親 剛 開學 考試 你 怎么 就 得 個 ‘ 0 ’ 分 ? 兒子 老師 說 我們 一切 都 要 從 ‘ 0 ’ 開始'
#                   ]
#XX = np.empty(len(INPUT_SENTENCES),dtype=list)
## 轉換文字為數值
#i=0
#for sentence in  INPUT_SENTENCES:
#    words_temp = nltk.word_tokenize(sentence.lower())
#    seq = []
#    for word in words_temp:
#        if word in word2int:
#            seq.append(word2int[word])
#        else:
#            seq.append(word2int['UNK'])
#    XX[i] = seq
#    i+=1
#
#XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
## 預測，並將結果四捨五入，轉換為 0 或 1
#labels = [int(round(x[0])) for x in model.predict(XX) ]
#label2word = {1:'是笑話-', 0:'不是笑話-'}
## 顯示結果
#for i in range(len(INPUT_SENTENCES)):
#    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))  
    
#-----------------------------------------------------------------------#    
            
#words=[]
#for word in model_w2v.wv.vocab:
#    words.append(word)
#words = set(words)
    
#model = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)
#vectors = model.fit_transform(vectors)
#normalizer = preprocessing.Normalizer()
#vectors =  normalizer.fit_transform(vectors, 'l2')
#
#fig, ax = plt.subplots()
#for word in words:
##    print(word,vectors[word2int[word]][0] ,vectors[word2int[word]][1])
#    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
##        print(word)
#plt.xlim((-1, 1))
#plt.ylim((-1, 1))
#plt.show()