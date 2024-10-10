# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:01:41 2018

@author: 蒲立年
"""

from bs4 import BeautifulSoup
import requests
import codecs
url = "https://forum.gamer.com.tw/C.php?bsn=60076&snA=4544033"
sitepage = requests.get(url)
sitepage.encoding = 'utf8'
soup = BeautifulSoup(sitepage.text,'lxml')
joke_str=soup.select('div.c-article__content')[0].text.strip()
#print(joke_str)
writing=0
onejoke=""
num=0
for i in range(len(joke_str)):
    if joke_str[i]=='=':
        if writing==1:
            f.write('{}'.format(onejoke))
            onejoke=""
            f.close()
            writing=0
        continue
    if writing==0:
        temp='joke'+str(num)+'.txt'
        num+=1
        f=codecs.open(temp, 'w','utf8')
        writing=1
    onejoke+=joke_str[i]
#    print(joke_str[i])
#    print(onejoke)
   