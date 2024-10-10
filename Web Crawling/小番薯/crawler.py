# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

  
from bs4 import BeautifulSoup
import requests
import codecs
import subprocess
url = "http://kids.yam.com/joke/article.php?cid=campus&id=174334" #小番薯網站
for i in range(2):
    temp='txt'+str(i)+'.txt'
    f=codecs.open(temp, 'w')
    sitepage = requests.get(url)
    sitepage.encoding = 'big5'
    soup = BeautifulSoup(sitepage.text,'lxml')
    joke_str=soup.select('td.tableword2 div')[0].text.strip()
#    funny_index=soup.select('.t10c3')[0].text #爆笑指數
#   ques_test(reup)    
    #print(reup.select('.board')[0])
#    f.write('{}\n{}'.format(funny_index,joke_str) ) 
    f.write('{}'.format(joke_str) ) 
    next=soup.select('.cl')[8]
    href=next['href'].lstrip('.')
#    href.lstrip('/')
    url = "http://kids.yam.com/joke"
    url+=href
    #print(f.read())
    f.close()
    change_temp='ch_'+temp
    command_="CKIPWSTester.exe ws.ini "+temp+" "+change_temp
#    print(command_)
    process = subprocess.Popen(command_, stderr=subprocess.PIPE)
    print("end of subprocess call "+temp)
    
    
    
