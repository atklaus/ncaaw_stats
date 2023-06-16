import requests
from lxml import html
import os
import pandas as pd
import html5lib
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import lxml.html as lh
import importlib
from dotenv import load_dotenv
from pathlib import Path
import string
letters=list(string.ascii_lowercase)
import utils
importlib.reload(utils)
import pandas as pd
import requests
import random

dotenv_path = Path('/Users/adamklaus/.env')
load_dotenv(dotenv_path=dotenv_path)

# https://free-proxy-list.net/

BASE_URL = 'https://www.basketball-reference.com/wnba/players/{}/'
REF_HOME = 'https://www.basketball-reference.com'

proxies = open("proxies.txt", "r").read().strip().split("\n")

def get(url, proxy): 
	""" 
	Sends a GET request to the given url using given proxy server. 
	The proxy server is used without SSL, so the URL should be HTTP. 
 
	Args: 
		url - string: HTTP URL to send the GET request with proxy 
		proxy - string: proxy server in the form of {ip}:{port} to use while sending the request 
	Returns: 
		Response of the server if the request sent successfully. Returns `None` otherwise. 
 
	""" 
	try: 
		r = requests.get(url, proxies={"http": f"http://{proxy}"}) 
		if r.status_code < 400: # client-side and server-side error codes are above 400 
			return r 
		else: 
			print(r.status_code) 
	except Exception as e: 
		print(e) 
 
	return None

def check_proxy(proxy): 
	""" 
	Checks the proxy server by sending a GET request to httpbin. 
	Returns False if there is an error from the `get` function 
	""" 
 
	return get("http://httpbin.org/ip", proxy) is not None 
 
available_proxies = list(filter(check_proxy, proxies))

proxies = {'http': 'http://{}'} 
response = requests.get('http://httpbin.org/ip', proxies=proxies) 
print(response.json()['origin']) # 190.64.18.162

user_agents = [
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64; Trident/7.0; Touch; MASMJS; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 10.0; AOL 9.7; AOLBuild 4343.1028; Windows NT 6.1; WOW64; Trident/7.0)",
    "Mozilla/5.0 (Linux; U; Android 4.0.3; en-us) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.59 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Trident/7.0; Touch; TNJB; rv:11.0) like Gecko",
    "Mozilla/5.0 (iPad; CPU OS 8_1_3 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Mobile/12B466",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; Active Content Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.130 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.124 Safari/537.36",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; InfoPath.3)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0; WebView/1.0)",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.89 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.130 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.91 Safari/537.36",
    "Mozilla/5.0 (iPad; U; CPU OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.130 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) coc_coc_browser/50.0.125 Chrome/44.0.2403.125 Safari/537.36",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64; Trident/7.0; MAARJS; rv:11.0) like Gecko",
    "Mozilla/5.0 (Linux; Android 5.0; SAMSUNG SM-N900T Build/LRX21V) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/2.1 Chrome/34.0.1847.76 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 8_4 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) GSA/7.0.55539 Mobile/12H143 Safari/600.1.4"
]


proxies = ['45.119.208.134:80', '20.206.106.192:80', '46.122.0.246:80', '103.77.60.14:80', '3.36.130.175:80', '203.198.207.253:80', '143.244.182.101:80', '20.210.113.32:80', '149.28.25.61:80']

def requst_params(user_agents, available_proxies):    
    user_agent = random.choice(user_agents) 
    headers = {'User-Agent': user_agent} 
    proxy = random.choice(available_proxies) 
    proxies = {'http': 'http://{}'.format(proxy)} 
    return headers, proxies


pd.read_html('https://www.sports-reference.com/cbb/schools/#all_NCAAW_schools')

url = BASE_URL.format(letter)

headers, proxy_rand = requst_params(user_agents, available_proxies)
session = requests.session()

response = session.get(url, headers = headers, proxies=proxy_rand)
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = utils.get_url_dict(page_html)


for letter in letters:
    url = BASE_URL.format(letter)
    session = requests.session()

    response = session.get(url, headers = headers, proxies=proxy_rand)
    page_html = BeautifulSoup(response.text, 'html5lib')
    url_dict = utils.get_url_dict(page_html)

    player_urls = []
    for key, value in url_dict.items():
        if ('/wnba/players/{}/'.format(letter) in value) & ('/wnba/players/{}/'.format(letter) != value):
            player_urls.append(value)

    for player_url in player_urls:
        import time
        time.sleep(5.5)
        session = requests.session()

        player_url = REF_HOME + player_url
        headers, proxy_rand = requst_params(user_agents, available_proxies)
        response = session.get(player_url, headers = headers, proxies=proxy_rand)
        page_html = BeautifulSoup(response.text, 'html5lib')
        url_dict = utils.get_url_dict(page_html)
        div_class = page_html.findAll('h1')
        player_name = div_class[0].find('span').text

        tables = page_html.findAll("table")
        for table in tables:
            if 'Advanced' in str(table):
                player_adv_df = pd.read_html(str(table))[0]
                player_adv_df = player_adv_df.add_prefix('adv_')
            if 'Per Game' in str(table):
                player_pg_df = pd.read_html(str(page_html))[0]
                player_pg_df = player_pg_df.add_prefix('pg_')

        base_df = player_pg_df.merge(player_adv_df,how='left',left_on='pg_Year',right_on='adv_Year')
        base_df['player_name'] =player_name

        for key, value in url_dict.items():
            if 'college=' in value:
                base_df['college_team'] = key

        base_df.to_csv('wnba_ref/' + player_name + '.csv')


BASE_URL = 'https://www.sports-reference.com/'
team_url = 'https://www.sports-reference.com/cbb/schools/abilene-christian/women/'


player_url = REF_HOME + player_url
response = session.get(team_url, headers = headers)
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = utils.get_url_dict(page_html)
team_page_df = pd.read_html(str(page_html),header=1)[0]
team_page_df['url'] =BASE_URL + team_page_df['Season'].map(url_dict)




div_class = page_html.findAll('h1')
player_name = div_class[0].find('span').text



tables = page_html.findAll("table")
for table in tables:
    if 'Advanced' in str(table):
        player_adv_df = pd.read_html(str(table))[0]
        player_adv_df = player_adv_df.add_prefix('adv_')
    if 'Per Game' in str(table):
        player_pg_df = pd.read_html(str(page_html))[0]
        player_pg_df = player_pg_df.add_prefix('pg_')