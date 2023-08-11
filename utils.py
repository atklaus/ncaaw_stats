import requests
from lxml import html
import os
import pandas as pd
import html5lib
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import lxml.html
import importlib
import json
import fuzzywuzzy
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

#Pull all W players
# https://www.basketball-reference.com/friv/colleges.fcgi?college=augustast

os.chdir('/Users/adamklaus/Documents/Personal/Develop/ncaaw_stats')
from constants import LOGIN_URL, HOME_URL, PRO_HOME_URL, NCAA_TEAMS_URL, TABLE_CLASS, CREDS_DICT, PLAYER_INPUT_DICT

def login():
    '''
    returns a session that is logged into herhoopstats.com
    '''
    s = requests.session()
    login = s.get(LOGIN_URL)
    login_html = lxml.html.fromstring(login.text)
    hidden_inputs = login_html.xpath(r'//form//input[@type="hidden"]') #find any hidden attributes that must be included in the post
    form = {x.attrib["name"]: x.attrib["value"] for x in hidden_inputs}
    print(form)
    # {'csrftok': '9e34ca7e492a0dda743369433e78ccf10c1e68bbb1f453cbb80ce6eaeeebe928', 
    #  'context': ''}
    form['email'] = CREDS_DICT['username']
    form['password'] = CREDS_DICT['password']
    form['Connection'] = 'keep-alive'

    response = s.post(LOGIN_URL, data=form, headers = dict(referer=LOGIN_URL))
    return s


def get_table_by_elm_text(page_html, find_text, element, class_name):

    table_df = None
    div_class = page_html.findAll('div',class_=class_name)

    for div in div_class:
        try:
            if find_text == div.find(element).text:
                get_div = div
                table_df = pd.read_html(str(get_div))
        except:
            print('Element does not exist in this div')

    if table_df == None:
        print('WARNING: No data in return df')

    return table_df[0]


def get_url_dict(page_html):
    '''
    Create a dictionary of all links and their text reference
    '''
    href = page_html.find_all('a')
    href_dict = {}
    for item in href:
        try:
            href_dict[item.text.strip()] = item['href']
        except:
            pass
    
    return href_dict

def get_html(session,url):
    response = session.get(url, headers = dict(referer = url))
    page_html = BeautifulSoup(response.text, 'html5lib')
    return page_html

#Write dictionary to json
def write_dict_to_json(write_dict,filepath):
    js = json.dumps(write_dict)
    # Open new json file if not exist it will create
    fp = open(filepath, 'a')
    # write to json file
    fp.write(js)
    # close the connection
    fp.close()


def read_json(filepath):
    with open(filepath) as f:
        read_dict = json.load(f)
    return read_dict



# player = page_html.find_all(id='player_stats')


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

# https://proxyscrape.com/free-proxy-list 
# available_proxies = list(filter(check_proxy, proxies))
available_proxies = open("proxies.txt", "r").read().strip().split("\n")

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



def requst_params(user_agents, available_proxies):    
    user_agent = random.choice(user_agents) 
    headers = {'User-Agent': user_agent} 
    proxy = random.choice(available_proxies) 
    proxies = {'http': 'http://{}'.format(proxy)} 
    return headers, proxies


def extract_awards(page_html):
    """Extract awards from the provided HTML."""
    html_list = []
    try:
        html_list = page_html.find('ul', id='bling')
        return ','.join([li.text for li in html_list.find_all('li')])
    except:
        return html_list

def extract_name_and_position(page_html):
    """Extract name and position from the provided HTML."""
    name = None
    position = None
    div_class = page_html.find('div', class_='nothumb')
    try:
        name = div_class.find('span').text
    except:
        pass 
    try:
        position = div_class.find_all('p')[0].text.split(':')[1].strip()    
    except:
        pass

    return name, position

def extract_height(page_html):
    """Extract and format height from the provided HTML."""
    try:
        div_class = page_html.find('div', class_='nothumb')
        height = div_class.find_all('p')[1].text.strip()
        match = re.search(r'\((.*?)\)', height)
        
        if match:
            height = match.group(1)  # group(1) corresponds to the first group enclosed in parentheses
            return height.replace('cm','').strip()
    except:
        return None

def extract_details_from_page(page_html):
    """Extract required details from the provided HTML."""
    try:
        awards = extract_awards(page_html)
        name, position = extract_name_and_position(page_html)
        height = extract_height(page_html)
        return awards, name,position,height
    except Exception as e:
        # Logging the exception might be helpful for debugging purposes.
        print(f"Error occurred: {e}")
        return {}


# import requests
# from bs4 import BeautifulSoup
# import time
# import random

# USER_AGENTS = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     # Add more user agents if needed
# ]

# def get_random_user_agent():
#     return random.choice(user_agents)


# def is_proxy_valid(proxy):
#     url = "http://httpbin.org/ip"
#     proxies = {
#         "http": proxy,
#         "https": proxy
#     }
#     try:
#         response = requests.get(url, proxies=proxies, timeout=5, headers={"User-Agent": get_random_user_agent()})
#         if response.status_code == 200:
#             return True
#     except requests.RequestException:
#         pass
#     return False

# def get_valid_proxies():
#     all_proxies = get_free_proxies()
#     valid_proxies = [proxy for proxy in all_proxies if is_proxy_valid(proxy)]
#     return valid_proxies

# # Introduce delay before making a request
# time.sleep(random.uniform(1, 5))

# # Example usage
# valid_proxies = get_valid_proxies()
# print(valid_proxies)
