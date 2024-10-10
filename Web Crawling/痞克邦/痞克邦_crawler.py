# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:45:00 2018

@author: 蒲立年
"""

from bs4 import BeautifulSoup
import requests
import codecs
url = "http://csf8899.pixnet.net/blog/post/55525306-131%E5%80%8B%E7%AC%91%E8%A9%B1%EF%BC%8C%E7%AC%91%E5%80%8B%E5%A4%A0%EF%B9%97"
sitepage = requests.get(url)
sitepage.encoding = 'utf8'
soup = BeautifulSoup(sitepage.text,'lxml')
joke_str=soup.select('p span')[0].text.strip()
writing=1
onejoke=""
num=0
f=codecs.open('pre.txt', 'w','utf8')
for i in range(len(joke_str)):
    if joke_str[i]>='0' and joke_str[i]<='9':
        if writing==1:
            f.write('{}'.format(onejoke))
            onejoke=""
            f.close()
            writing=0
        continue
    if joke_str[i]=='.':
        continue
    if writing==0:
        temp='joke'+str(num)+'.txt'
        num+=1
        f=codecs.open(temp, 'w','utf8')
        writing=1
    onejoke+=joke_str[i]
f.write('{}'.format(onejoke))
f.close()