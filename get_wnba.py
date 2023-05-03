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

dotenv_path = Path('/Users/adamklaus/.env')
load_dotenv(dotenv_path=dotenv_path)

url='https://www.basketball-reference.com/wnba/players/a/'
import utils
importlib.reload(utils)


page_html = utils.get_html(url)

session = requests.session()
response = session.get(url, headers = dict(referer = url))
page_html = BeautifulSoup(response.text, 'html5lib')

url_dict = utils.get_url_dict(page_html)
REF_HOME = 'https://www.basketball-reference.com'
player_url =REF_HOME + url_dict['Bella Alarie']

response = session.get(player_url, headers = dict(referer = player_url))
page_html = BeautifulSoup(response.text, 'html5lib')

season_df = utils.get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')