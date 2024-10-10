# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:44:29 2018

@author: LaynelMoon
"""

from bs4 import BeautifulSoup
import requests
import time
url_page = "https://www.ptt.cc/bbs/joke/index6223.html"
num=0
for i in range(1):
#    with open(temp,'w', encoding='utf-8') as f:
    sitepage = requests.get(url_page)
    sitepage.encoding = 'utf8'
    soup = BeautifulSoup(sitepage.text,'lxml')
    for j in soup.select('.title a'):
#        reup=soup.select('.title a')[j]
        url_section='https://www.ptt.cc'+j['href']
        print(url_section)
        sitepage2 = requests.get(url_section)
        sitepage2.encoding = 'utf8'
        soup2 = BeautifulSoup(sitepage2.text,'lxml')
        [s.extract() for s in soup2(['span'])] #移除特定標籤
        content=soup2.select('#main-content')
#        print(contents)
#        for content in contents:
#            print(content)
        text=content[0].get_text().strip().rstrip('-').strip()

        sentence=""
        sentfrom=0
        for word in text:
            if sentfrom==1:
                if word=="\n":
                    sentfrom==0
            else:
                if word=="S" or word=="-" or word=="h":
                    sentfrom=1
                else:
                    sentence+=word
        temp='text'+str(num)+'.txt'
        if sentence != "":
            with open(temp,'w+', encoding='utf-8') as f:
                f.write('{}'.format(sentence))
            num+=1
    next_url=soup.select('a:nth-of-type(8)')[0]['href']
    url_page="https://www.ptt.cc"+next_url
#        print(href)