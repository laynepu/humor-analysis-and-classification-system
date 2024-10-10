# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:59:56 2018

@author: LaynelMoon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:44:29 2018

@author: LaynelMoon
"""

from bs4 import BeautifulSoup
import requests
import time
url_page = "http://ibook.idv.tw/enews/Survey.html"

headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
num=-1
#    with open(temp,'w', encoding='utf-8') as f:
sitepage = requests.get(url_page)
sitepage.encoding = 'big5'
soup = BeautifulSoup(sitepage.text,'html.parser') #lxml找不到全部
#[s.extract() for s in soup(['p'])]
#print(soup)
#time.sleep(10)
i=0
#print(soup.select('table tr td a:nth-of-type(1)'))
#print(soup.select('table tr td:nth-of-type('+str(17+num)+')')[0].get_text())
for k,j in enumerate(soup.select('table tr td a')):
    if k == 0:
        continue
    print(i)
    is_joke=-1
    try:
        while 1:
            num+=1
            label=soup.select('table tr td:nth-of-type('+str(16+num)+')')[0].get_text() #第一個數字位置在16
            if label.isdigit():
#                print('is_digit')
                label=soup.select('table tr td:nth-of-type('+str(16+num+1)+')')[0].get_text()
                #lebal一定在數字後一個
                break
        print(label)
    except IndexError:
        print("笑話已全數抓取完畢!!")
        break
    if label == "圖文":
        continue
    if label.find("笑話") != -1:
        is_joke=1
#    time.sleep(1)
#        reup=soup.select('.title a')[j]
#    url_section='http://ibook.idv.tw/enews/enews1-30/enews14.html' #font多一個
    url_section='http://ibook.idv.tw/enews/'+j['href']
    sitepage2 = requests.get(url_section)
    sitepage2.encoding = 'big5'
    soup2 = BeautifulSoup(sitepage2.text,'lxml')
#        print(soup2)
#        [s.extract() for s in soup2(['span'])] #移除特定標籤
#    contents=soup2.select('font:nth-of-type(10)')
#    contents=soup2.select('table td:nth-of-type(8)')
#    [s.extract() for s in soup2(['script','h1'])]
#    print(contents[0].get_text().strip().replace(' ',''))
#    time.sleep(8)
    try:
        for n in range(5):
            contents=soup2.select('table td:nth-of-type('+str(n+4)+')')
            [s.extract() for s in soup2(['script','h1'])]
            text=contents[0].get_text().strip().replace(chr(160),'').replace(' ','').replace('　','') #移除字串中的空格，chr(160)為一種特殊空格
            if len(text) > 25:
                break
#        print('text=',text)
        time.sleep(0.5)
        if text == "" or text.find("報長的話") != -1:
            continue
#        text=contents[0].get_text().strip().replace(' ','') #移除字串中的空格
#        elif len(text) < 10:
#            contents=soup2.select('table td:nth-of-type(6)')
#            [s.extract() for s in soup2(['script'])]
#            print(contents)
#        text=contents[0].get_text().strip().replace(' ','') #移除字串中的空格
        print('text=',text)
#        time.sleep(10)
    except IndexError:
        print(contents)
        continue
#    print(contents)
#    time.sleep(10)
#    [s.extract() for s in soup2(['br'])] #移除特定標籤
#    print(contents)
#    time.sleep(10)
#        for content in contents:
#            print(content)
    
    temp='text'+str(i)+'.txt'
    with open(temp,'w+', encoding='utf-8') as f:
        if is_joke == 1:
            f.write('1\n')
        else:
            f.write('0\n')
        f.write('{}'.format(text))
    i+=1
#    next_url=soup.select('a:nth-of-type(8)')[0]['href']
#    url_page="https://www.ptt.cc"+next_url
##        print(href)
print("程式結束")