# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:16:24 2018

@author: LaynelMoon
"""
from keras.models import load_model
from keras.preprocessing import sequence
from gensim.models import word2vec
import numpy as np
import nltk
import keras.backend as K
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall  


model = word2vec.Word2Vec.load("gensim_word2vec_學習電子報+zol笑話大全_data.model")
num_recs = 0
with open('./學習電子報+zol笑話大全_data.txt','r+', encoding='UTF-8') as f:
    for line in f:
        num_recs += 1

EMBEDDING_SIZE = 10
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
words=[]
for index,word in enumerate(model.wv.vocab):
    words.append(word)
    word2int[word]=index+2
    vector.append(model.wv[word])
    vocab_size+=1
words = set(words)
index2word = {v:k for k, v in word2int.items()}
vectors=np.array(vector)

model = load_model('keras_LSTM_gensim_sg_學習電子報+zol笑話大全_data.h5', custom_objects={'precision':precision, 'recall':recall})
data_size=122
correct_num=0
INPUT_SENTENCES=[]
for i in range(data_size):
    label=-1;
    INPUT_SENTENCES=[]
    with open('./text_seg'+str(i)+'.txt', 'r+', encoding='UTF-8') as p:
        for j,line in enumerate(p):
            if j == 0:
                if line == '1\n':
                    label=1
                else:
                    label=0
                continue
            INPUT_SENTENCES.append(line)
    if label == -1:
        print(i,'ERROR')
#INPUT_SENTENCES = [ '用於 說服 反對 建造 新 主力艦 的 閣員 '
#                   ,'不過 由於 整個 銀河系 的 戰況 吃緊'
#                   ,'老婆 說 嫁給 魔鬼 也 比 嫁給 你好 老公 說 不行 啊 近親 不能 結婚 啊'
#                   ,'秝瑋 要 去 睡覺 了'
#                   ,'我餓了，想要去吃飯了'
#                   ,'老師 誰 的 優點 最 多 甲 大華 老師 為 什麼 甲 因為 他家 開 藥局 '
#                   , '父親 剛 開學 考試 你 怎么 就 得 個 ‘ 0 ’ 分 ? 兒子 老師 說 我們 一切 都 要 從 ‘ 0 ’ 開始'
#                   ]
    XX = np.empty(len(INPUT_SENTENCES),dtype=list)
    # 轉換文字為數值
    index=0
    for sentence in  INPUT_SENTENCES:
        words_temp = nltk.word_tokenize(sentence.lower())
        seq = []
        for word in words_temp:
            if word in word2int:
                seq.append(word2int[word])
            else:
                seq.append(word2int['UNK'])
        XX[index] = seq
        index+=1
    
    MAX_SENTENCE_LENGTH = 40
    XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    print("-------------------------------------------------------")
    print("文章{}".format(i))
    print('好笑程度      句子')
    # 預測，並將結果四捨五入，轉換為 0 或 1
    humor_degree=0
    for i,x in enumerate(model.predict(XX)) :
        print('{0:.8f}   {1}'.format(x[0], INPUT_SENTENCES[i]))
        humor_degree+=x[0]
    humor=humor_degree/len(INPUT_SENTENCES)
    #labels = [int(round(x[0])) for x in model.predict(XX) ]
    label2word = {1:'是笑話-', 0:'不是笑話-'}
    is_joke=int(round(humor))
    if is_joke == label:
        correct_num+=1
    # 顯示結果
    #for i in range(len(INPUT_SENTENCES)):
    print('-->實際:{0} 預測:{1} 好笑程度: {2:.3f}'.format(label2word[label],label2word[is_joke], humor))  
print('預測成功率 : {0:.1f}%，笑話數量 : {1}'.format(correct_num*100/data_size, correct_num))