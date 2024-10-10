# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:08:01 2018

@author: 蒲立年
"""

from bs4 import BeautifulSoup
import requests
import codecs
import math
import time
from opencc import OpenCC
cc=OpenCC('s2t')
#proxies = {
#        'http': 'socks5://127.0.0.1:21',
#        'https': 'socks5://127.0.0.1:21'
#}
#sitepage = requests.get('https://www.mdnkids.com/mdnjoke/', proxies=proxies)
#net_num=1
num=9437
headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
for k in range(10000,20000):
    url = "https://xiaohua.zol.com.cn/detail60/"+str(k)+".html"
    try:
        sitepage = requests.get(url,headers=headers)
    except ConnectionError:
        print("detail60/",k,"連結錯誤")
        continue
    sitepage.encoding = 'gbk'
    soup = BeautifulSoup(sitepage.text,'lxml')
    temp='joke'+str(num)+'.txt'
    num+=1
    index=1
    f=codecs.open(temp,"w","utf8") #codecs能存入unicode編碼字符
    onejoke=""
#    for i in range(1):
    try:
        joke_str=soup.select('.article-text')[0].text.strip()# span:nth-of-type('+str(index)+')
        joke_str=''.join(joke_str.split())
        joke_str=cc.convert(joke_str)
        for i in range(len(joke_str)):
            onejoke+=joke_str[i]
        print(onejoke)
    #            
    #            index+=2
    #        except IndexError:
    #            try:
    #                print('結束')
    #                print(onejoke)
        f.write(onejoke)
        print('成功存入joke',num-1,".txt"," k=",k)
    
#    f.close()
    except UnicodeEncodeError:
        num-=1
        print("此篇含有Unicode編碼問題")
    except IndexError:
        print("detail60/",k,"不存在")
        num-=1
    f.close()
#    net_num+=1
#                num-=1連線失敗
    time.sleep(0.5)
#            break;
#    try:
#        next=soup.select('a.next')[0]
#        href="https://xiaohua.zol.com.cn"+next['href'] #https://xiaohua.zol.com.cn
#        url=href
#    except IndexError:
#        print("錯誤:",num,next)
    