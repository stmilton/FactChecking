import requests
import re
from bs4 import BeautifulSoup

# craw html from url(list) and return web text with a string(list)
# the result text is divided with ".", without handling more punctuations like "，", "。", "[]"
def url_to_text(url):
    # the API only use these parameters
    JSON = {
        "urls": url,
        "cache": False,
        "timeout": 15000
    }
    # use API
    web = requests.post("http://140.115.54.45:6789/post/crawler/static/html", json=JSON)

    # filter the content of return string(list)
    web_list = web.json()
    return_list = []
    for element in web_list:
        soup = BeautifulSoup(element, 'html.parser')
        temp = soup.get_text()
        dr = re.compile(r'(\t)+(\n)*|(\t)*(\n)+')
        temp = dr.sub('.',temp)
        dr = re.compile('[.]+')
        temp = dr.sub('.',temp)
        return_list.append(temp)

    return return_list