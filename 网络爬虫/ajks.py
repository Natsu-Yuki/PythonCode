import requests
from pyquery import PyQuery as pq
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import json
import re
from urllib.parse import urlencode
data={
    'offset': '0',
    'format': 'json',
    'keyword': '街拍',
    'autoload': 'true',
    'count': '20',
    'cur_tab': '1',
    'from': 'search_tab'
}

url='https://www.toutiao.com/search_content/?'+urlencode(data)
print(url)
html=requests.get(url).text
data=json.loads(html)
urls=[]
for  i in data.get('data'):
    if i.get('article_url')==None:
        pass
    else:urls.append(i.get('article_url'))

print(urls[0])
# html=requests.get(urls[0]).text
#
# path=r'C:\Users\Natsu\PycharmProjects\untitled\WORKSPACE\a.html'
# with open(path,'w') as f:
#     f.write(html)
# pattern=re.compile('JSON.parse(.*)',re.S)
# result=re.search(pattern,html)
# print(result)


# https://www.toutiao.com/search_content/?offset=0&format=json&keyword=%E8%A1%97%E6%8B%8D&autoload=true&count=20&cur_tab=1&from=search_tab
# url='https://api.bilibili.com/x/web-interface/archive/stat?jsonp=jsonp&aid=3565752'
# user_agent = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
# headers = {"User-Agent": user_agent}
# r=requests.get(url,headers=headers).json()




