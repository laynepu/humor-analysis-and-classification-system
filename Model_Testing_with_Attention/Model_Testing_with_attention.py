# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:16:24 2018

@author: LaynelMoon
"""
from keras.models import load_model
from keras.engine.topology import Layer
from keras.utils import CustomObjectScope
from keras import initializers, regularizers, constraints
from keras.preprocessing.sequence import pad_sequences
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

model = word2vec.Word2Vec.load("gensim_word2vec_學習電子報+zol笑話大全_data_for_model_training_with_attention.model")
num_recs = 0
with open('./學習電子報+zol笑話大全_data_for_model_training.txt','r+', encoding='UTF-8') as f:
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
with CustomObjectScope({'AttentionWithContext': AttentionWithContext(),'precision': precision,'recall': recall}):
    model = load_model('學習電子報+zol笑話大全_data_for_model_training_with_attention.h5')
#model = load_model('keras_LSTM_gensim_sg_學習電子報+zol笑話大全_data_for_model_training.h5', custom_objects={'AttentionWithContext': AttentionWithContext(),'precision':precision, 'recall':recall})
data_size=122
correct_num=0
joke_sum=0
INPUT_SENTENCES=[]
for i in range(data_size):
#for i in range(data_size):
    
    label=-1;
    INPUT_SENTENCES=[]
    try :
        with open('./testing_data_痞克邦/text_seg'+str(i)+'.txt', 'r+', encoding='UTF-8') as p:
            for j,line in enumerate(p):
                if j == 0:
                    if line == '1\n':
                        label=1
                        joke_sum=joke_sum+1
                    else:
                        label=0
                    continue
                INPUT_SENTENCES.append(line)
    except FileNotFoundError:
#        print('text_seg'+str(i)+'.txt','不存在\n')
        continue
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
    
    MAX_SENTENCE_LENGTH = 16
    XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH,padding='post',truncating='post')
    
    
    pos_key=[]  #attention
    pos_value=[]
    neg_key=[]
    neg_value=[]
    Knum = 3
    weight=np.array(model.get_weights())
    for X in XX:
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
        test_seq = pad_sequences([X], maxlen=MAX_SENTENCE_LENGTH,padding='post',truncating='post')
        out = get_layer_output([test_seq, 0])[0]  # test mode
        eij = np.tanh(np.dot(out[0], weight[4][0]))
        ai = np.exp(eij)
        weights = ai/np.sum(ai)
        topKeys = np.argpartition(weights,-Knum)[-Knum:]
        for keys in topKeys:
            if(model.predict(test_seq)>0.4):
                pos_key.append(X[keys])
                pos_value.append(weights[keys])
            else:
                neg_key.append(X[keys])
                neg_value.append(weights[keys])
    pos_key=np.array(pos_key)
    pos_value=np.array(pos_value)
    neg_key=np.array(neg_key)
    neg_value=np.array(neg_value)
    if pos_value.size<5:
        topKeys = np.argpartition(pos_value,-3)[-3:]
    else:
        topKeys = np.argpartition(pos_value,-5)[-5:]
    print("------")
    for keys in topKeys:
        print(index2word[pos_key[keys]],pos_key[keys],pos_value[keys])
    if neg_value.size<5:
        topKeys = np.argpartition(neg_value,-3)[-3:]
    else:
        topKeys = np.argpartition(neg_value,-5)[-5:]
    print("------")
    for keys in topKeys:
        print(index2word[neg_key[keys]],neg_key[keys],neg_value[keys])
    
    
    
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
    is_joke=int(round(humor+0.1))
    if is_joke == label:
        correct_num+=1
    # 顯示結果
    #for i in range(len(INPUT_SENTENCES)):
    print('-->實際:{0} 預測:{1} 好笑程度: {2:.3f}'.format(label2word[label],label2word[is_joke], humor))  
print('預測成功率 : {0:.1f}%'.format(correct_num*100/data_size))