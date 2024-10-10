"""
Spyder Editor

This is a temporary script file.
"""
#encoding=utf-8
  
import jieba
import jieba.analyse
import codecs
import logging
joke_num= input("起始笑話目錄?")
dic={}

stopword_set=set()
with open('stop_words.txt','r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))
data=codecs.open('痞克邦笑話_data.txt','w+',encoding='utf-8')
for i in range(122):
#    print(int(joke_num))
    print(int(joke_num)+i)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    jieba.set_dictionary('dict.txt.big')
#    jieba.set_dictionary('dict.txt.small')
    temp='joke'+str(i+int(joke_num))+'.txt'
    try:
        joke=open(temp,'rb')
#        alphabet=open(temp1,'a')
        content=joke.read()
#        print("Input:", content)
        words=jieba.cut(content, cut_all=False)
#        print("Output 精確模式 Full Mode")
#        好笑
#        alphabet.write('1\t')
#        不好笑
#        alphabet.write('0\t')
        word_count=0
        for word in words:
            if word not in stopword_set:
#                print(word,end="/")
                if word!='\r\n'and word!='\r' and word!='\n':
#                    if word in dic:
#                        dic[word]+=1
#                    else:
#                        dic[word]=1
#                    print(word+'/')
                    if word_count==0:
                        if word!='？' and word!='。' and word != '!' and word != '！' and word!='\r\n'and word!='\r' and word!='\n' and word!='：':
                            data.write('1\t')
                            data.write(word+' ')
                            word_count=1
                            line=True;
                    else:
                        if word=='？' or word=='。' or word == '!' or word == '！' or word_count>15 or word=='\r\n'or word=='\r' or word=='\n' or word=='：':
                            if line:
                                data.write('\n')
                                if word_count>15:
                                    data.write('1\t'+word+' ')
                                    word_count=1
                                    line=True
                                else:
                                    word_count=0
                                    line=False
                        else:
                            data.write(word+' ')
                            word_count+=1
                            line=True
        if line:
            data.write('\n')
        joke.close()
#        num+=1
            
    except FileNotFoundError:
        print(temp,'不存在\n')
data.close()
#    finally:
#        .close()
#alphabet.close()
#    print(type(words))
#    temp='word'+str(i+int(joke_num))+'.txt'
#    word=open(temp,'')
#print(str(dic)+'\n')
#print(max(dic, key=dic.get))
#print(dic[max(dic, key=dic.get)])

    