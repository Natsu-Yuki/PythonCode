import requests
from pyquery import PyQuery as pq
from bs4 import BeautifulSoup
import re
import pandas as pd
url='http://www.win4000.com/zt/meinv.html'
user_agent = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
headers = {"User-Agent": user_agent}
html=requests.get(url,headers=headers).text

soup=BeautifulSoup(html,'lxml')

items=soup.find_all('a',{'href':re.compile('.*'),'alt':re.compile('.*'),'title':re.compile('.*')})

row=0

for i in items:
    row+=1

    url_=i.get('href')

    html=requests.get(url_,headers=headers)
    soup=BeautifulSoup(html.text,'lxml')
    col=0
    for j in soup.find_all('img',{'alt':i.p.string,'data-original':re.compile('.*'),'src':re.compile('.*')}):
        h=requests.get(j.parent.get('href')).text
        s=BeautifulSoup(h,'lxml')
        r=requests.get(s.find('img',{'class':'pic-large'}).get('src'))
        col+=1
        path = r'C:\Users\Natsu\Desktop\a'+'\\'+i.p.string+'-'+str(row)+'-'+str(col)+'.jpg'
        with open(path,'wb') as f:
            f.write(r.content)