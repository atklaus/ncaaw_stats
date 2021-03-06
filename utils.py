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

