import pymysql as pms
db=pms.connect('localhost','root','123456','mydatabase',charset="utf8")
cursor = db.cursor()
url='http://maoyan.com/films'
import requests
from bs4 import BeautifulSoup
import pandas as pd
html=requests.get(url).text
soup=BeautifulSoup(html,'lxml')
titles=[]
for i in soup.find_all('div',{'class':'channel-detail movie-item-title'}):
    titles.append(i.get('title'))
scores=[]
for i in soup.find_all('div',{'class':'channel-detail channel-detail-orange'}):
    score=''
    for j in i.find_all('i'):
       score=score+j.string
    if score=='':
        score='暂无评分'
        scores.append(score)
    else:
        scores.append(score)

for t,s in zip(titles,scores):
   sql='insert into maoyan value (\'{}\',\'{}\');'.format(t,s)
   print(sql)
   cursor.execute(sql)
   db.commit()
db.close()

data=pd.DataFrame(
    {
        'title':titles,
        'score':scores
    }
)

print(data)





# import matplotlib.pyplot as plt
# import matplotlib.image as img
# path=r'C:\Users\Natsu\Pictures\CNN\2-1.jpg'
# image=img.imread(path)
# plt.imshow(image)
# plt.show()

