from selenium import webdriver
import selenium.webdriver.common.by as by
from bs4 import BeautifulSoup
import time
url='https://bangumi.bilibili.com/22/'
chromedriver = r"C:\Users\Natsu\AppData\Local\Google\Chrome\Application\chromedriver.exe"
browser = webdriver.Chrome(chromedriver)
browser.implicitly_wait(10)
browser.get(url)
browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
time.sleep(5)
html=browser.page_source
browser.close()
# print(type(html))
path=r'C:\Users\Natsu\PycharmProjects\untitled\WORKSPACE\a.html'
with open(path,'r+',encoding='utf-8') as f:
    f.write(html)
soup=BeautifulSoup(html,'lxml')
