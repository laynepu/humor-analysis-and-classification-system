# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:55:58 2018

@author: LaynelMoon
"""
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.models import load_model
from keras import initializers, regularizers, constraints
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import CustomObjectScope
#import nltk
#nltk.download('punkt')
#from sklearn import preprocessing
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from gensim.models import word2vec
#import time
import tensorflow as tf
import numpy as np
import keras.backend as K

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


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
 
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

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
#with open('./alphabet_sentence_new_equal.txt','r+', encoding='UTF-8') as f:
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
index2word = {v:k for k, v in word2int.items()}
vectors=np.array(vector)
print(vectors
      .shape)
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
#with open('./alphabet_sentence_new_equal.txt','r+', encoding='UTF-8') as f:
with open('./學習電子報+zol笑話大全_data_for_model_training.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words_temp = sentence.split()
#        print(words)
#        time.sleep(10)
        seqs = []
        for word in words_temp:
            if word in word2int:
                seqs.append(word2int[word])
            else:
                seqs.append(word2int["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
                    
                    
#保存模型，供日後使用
model_w2v.save("gensim_word2vec.model")
#模型讀取方式
# model = word2vec.Word2Vec.load("your_model_name")
    
MAX_SENTENCE_LENGTH = 16

# 字句長度不足補空白        
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH,padding='post',truncating='post') #大於maxlen會被截短，反之在後面補0

# 資料劃分訓練組及測試組
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# 模型構建
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 7
#model = Sequential()
##
## 加『嵌入』層
#model.add(Embedding(vocab_size, EMBEDDING_SIZE,weights=[vectors],input_length=MAX_SENTENCE_LENGTH))
#
## 加『LSTM』層
#model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(Dropout(0.5))
#model.add(AttentionWithContext())
#model.add(Dense(1))
#model.add(BatchNormalization())
#model.add(Activation("sigmoid"))
#
##precision = as_keras_metric(tf.metrics.precision)
##recall = as_keras_metric(tf.metrics.recall)
## binary_crossentropy:二分法
#model.compile(loss="binary_crossentropy", optimizer="adam",metrics=[precision, recall, "accuracy"])
##weight=np.array(model.get_weights())
##print(model.trainable_weights)
##print(weight.shape)
#model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))
### 模型存檔
#model.save('keras_LSTM_gensim_sg_equal_attention.h5')  # creates a HDF5 file 'model.h5'

# 模型訓練
#model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

#with CustomObjectScope({'AttentionWithContext': AttentionWithContext(),'precision': precision,'recall': recall}):
#    model = load_model('keras_LSTM_gensim_sg_equal_attention.h5')
with CustomObjectScope({'AttentionWithContext': AttentionWithContext(),'precision': precision,'recall': recall}):
    model = load_model('學習電子報+zol笑話大全_data_for_model_training.h5')
# 預測
#score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
#score,precision, recall, accuracy = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)

weight=np.array(model.get_weights())
for i in range(50):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
    test_seq = pad_sequences([Xtest[i]], maxlen=MAX_SENTENCE_LENGTH,padding='post',truncating='post')
    out = get_layer_output([test_seq, 0])[0]  # test mode
    print(out[0].shape)
    eij = np.tanh(np.dot(out[0], weight[4][0]))
    ai = np.exp(eij)
    weights = ai/np.sum(ai)
    Knum = 5
    print(weights.shape)
    print(weights)
    topKeys = np.argpartition(weights,-Knum)[-Knum:]
    print(topKeys)
    print(Xtest[i])
    print(ytest[i])
    print(model.predict(test_seq))
    for keys in topKeys:
        print(index2word[Xtest[i][keys]],Xtest[i][keys],weights[keys])


#print("\nTest score: %.3f, precision: %.3f, recall: %.3f, accuracy:%.3f" % (score, precision, recall, accuracy))
#print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
#print('{}   {}      {}'.format('預測','真實','句子'))
#for idx in range(len(Xtest)):
#    idx = np.random.randint(len(Xtest))
#    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
#    ylabel = ytest[idx]
#    ypred = model.predict(xtest)[0][0]
#    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
#    if int(round(ypred)) != int(ylabel):
#        print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
#    

    
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