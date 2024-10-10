# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:08:01 2018

@author: 蒲立年
"""

from bs4 import BeautifulSoup
import requests
import codecs
import math
#proxies = {
#        'http': 'socks5://127.0.0.1:21',
#        'https': 'socks5://127.0.0.1:21'
#}
#sitepage = requests.get('https://www.mdnkids.com/mdnjoke/', proxies=proxies)
url = "https://translate.googleusercontent.com/translate_c?depth=1&hl=zh-TW&prev=search&rurl=translate.google.com.tw&sl=zh-CN&sp=nmt4&u=https://xiaohua.zol.com.cn/detail60/59142.html&xid=17259,15700021,15700124,15700149,15700186,15700191,15700201,15700214&usg=ALkJrhj7w7aBAakNQ5smErRoYFD9U2Y4ow"
num=0
for k in range(5):
    sitepage = requests.get(url)
    sitepage.encoding = 'big5'
    soup = BeautifulSoup(sitepage.text,'lxml')
    temp='joke'+str(num)+'.txt'
    num+=1
    index=1
    f=open(temp,"w")
    onejoke=""
    while 1:
        try:
            joke_str=soup.select('.article-text span:nth-of-type('+str(index)+')')[0].text.strip()
            for i in range(math.ceil(len(joke_str)/2),len(joke_str)):
                onejoke+=joke_str[i]
                print(joke_str[i])
            index+=2
        except IndexError:
            print('結束')
            f.write(onejoke)
            f.close()
            break
    next=soup.select('a.next')[0]
    href=next['href']
    url=href