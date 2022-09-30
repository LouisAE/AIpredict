import csv
import re
import requests
import time
import os
import random
from bs4 import BeautifulSoup

headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.81 Safari/537.36 Edg/104.0.1293.47"}
csvHeaders=["日期","现汇买入价","现钞买入价","现汇卖出价","现钞卖出价"]
url="https://chl.cn/?"

def timeGet(lastDate:tuple=(2006,1,4)):
    L_T=[]
    Date=time.time()
    while True:
        T=time.gmtime(Date)
        L_T.append("%d-%d-%d"%T[:3])
        if T[:3]==lastDate:
            break
        Date-=86400
    return L_T[::-1]

def get(T,ID="usd",headers=headers):
    html=requests.get(url=url+T,headers=headers).text#中文爬虫编码解码典范了属于是
    html=html.encode("latin1").decode("utf-8")
    soup=str(BeautifulSoup(html,"html.parser"))
    p='<tr><td><a href="/\?%s">..</a></td><td>((\d)+.(\d)+)</td><td>(.+?)</td><td>((\d)+.(\d)+)</td><td>(.+?)</td></tr>'%ID
    M=re.findall(p,soup)
    target=re.findall(r'((\d)+.(\d)+)|-',str(M))
    target=[i[0] if i[0]!="" else "-" for i in target]
    target.insert(0,T)
    time.sleep(random.randint(0,5))
    return target

def write2f(address):
    with open(address,"w") as f:
        writer=csv.writer(f,delimiter="\t")
        writer.writerow(csvHeaders)
        for T in timeGet():
            writer.writerow(get(T,ID=ID))
            print("%s finish"%T)

def add2f(address):
    lastDate=None
    old=None
    with open(address,"r") as f:
        old=list(csv.reader(f,delimiter="\t"))
        lastDate=tuple(int(i) for i in old[-2][0].split("-"))
    with open(address,"a") as f:
        writer=csv.writer(f,delimiter="\t")
        for T in timeGet(lastDate):
            writer.writerow(get(T,ID=ID))
            print("%s finish"%T)

if __name__=="__main__":
    #ID=input("?:")
    #ID="gbp"
    ID="usd"
    address="./data/Analysis/%s.csv"%ID
    if os.path.exists(address):
        add2f(address)
    else:
        write2f(address)
    #print(get(url+"2006-1-4"))
