#!/bin/python3
from json import loads
from requests import get
from time import ctime

def getExchangeRate():
    res = loads(get("https://api.exchangerate-api.com/v4/latest/CNY").content.decode())
    return (res["time_last_updated"],res["USD"])
