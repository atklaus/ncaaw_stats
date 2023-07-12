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

# https://free-proxy-list.net/

base = 'https://www.sports-reference.com/cbb/players/{}-1.html'

wnba_df = pd.read_csv('all_wnba_players.csv')
wnba_df['name_url'] = wnba_df['name'].str.replace(' ',"-")
wnba_df['name_url'] = wnba_df['name_url'].str.replace("'","")
wnba_df['name_url'] = wnba_df['name_url'].str.lower()

# Append the url base to the 'name_url' column and store the result in a new column
wnba_df['url'] = wnba_df['name_url'].apply(lambda x: base.format(x))

player_urls = ['https://www.sports-reference.com/cbb/players/skylar-diggins-1.html','https://www.sports-reference.com/cbb/players/avery-warley-1.html','https://www.sports-reference.com/cbb/players/nnemkadi-ogwumike-1.html']

for player_url in list(wnba_df['url']):
# for player_url in player_urls:
    import time
    time.sleep(5)
    session = requests.session()
    headers, proxy_rand = utils.requst_params(utils.user_agents, utils.available_proxies)
    response = session.get(player_url, headers = headers, proxies=proxy_rand)
    # response = session.get(player_url)
    if response.status_code != 200:
        print(response.status_code)
    else:
        pass

    try:
        page_html = BeautifulSoup(response.text, 'html5lib')

        try:
            html_list = page_html.find('ul', id='bling')
            awards = ','.join([li.text for li in html_list.find_all('li')])
            div_class =page_html.find('div', class_='nothumb')
            name = div_class.find('span').text
            position = div_class.find_all('p')[0].text.split(':')[1].strip()
            height = div_class.find_all('p')[1].text.strip()
            match = re.search(r'\((.*?)\)', height)
            if match:
                height = match.group(1)  # group(1) corresponds to the first group enclosed in parentheses
                height = height.replace('cm','')
            else:
                height = 'N/A'
        except:
            pass


        div_class = page_html.findAll('h1')
        player_name = div_class[0].find('span').text

        tables = page_html.findAll("table")
        for table in tables:
            if 'Advanced' in str(table):
                player_adv_df = pd.read_html(str(table))[0]
                player_adv_df = player_adv_df.add_prefix('adv_')
                break
        for table in tables:
            if 'Per Game' in str(table):
                player_pg_df = pd.read_html(str(page_html))[0]
                player_pg_df = player_pg_df.add_prefix('pg_')
                break

        base_df = player_pg_df.merge(player_adv_df,how='left',left_on='pg_Season',right_on='adv_Season')
        base_df['player_name'] =player_name
        base_df['position'] =position
        base_df['height'] =height
        base_df['awards'] =awards

        base_df.to_csv('ncaa_ref/' + player_name + '.csv')
    except Exception as error:
    # handle the exception
        print("ERROR:", error)
        print(player_url)


# BASE_URL = 'https://www.sports-reference.com/'
# team_url = 'https://www.sports-reference.com/cbb/schools/abilene-christian/women/'


# player_url = REF_HOME + player_url
# response = session.get(team_url, headers = headers)
# page_html = BeautifulSoup(response.text, 'html5lib')
# url_dict = utils.get_url_dict(page_html)
# team_page_df = pd.read_html(str(page_html),header=1)[0]
# team_page_df['url'] =BASE_URL + team_page_df['Season'].map(url_dict)


# div_class = page_html.findAll('h1')
# player_name = div_class[0].find('span').text

# tables = page_html.findAll("table")
# for table in tables:
#     if 'Advanced' in str(table):
#         player_adv_df = pd.read_html(str(table))[0]
#         player_adv_df = player_adv_df.add_prefix('adv_')
#     if 'Per Game' in str(table):
#         player_pg_df = pd.read_html(str(page_html))[0]
#         player_pg_df = player_pg_df.add_prefix('pg_')

import requests
from bs4 import BeautifulSoup
import time

# Function to get the correct sports-reference link
def get_correct_link(player_name):
    # Replace spaces in player name with '+' for the Google search URL
    player_name_search = player_name.replace(' ', '+')
    
    # Google search URL
    url = f"https://www.google.com/search?q=site:sports-reference.com+{player_name_search}"

    # Send GET request
    response = requests.get(url)

    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        page_content = response.content

        # Create a Beautiful Soup object and specify the parser
        soup = BeautifulSoup(page_content, 'html.parser')

        # Find the link to the sports-reference page
        link = soup.find('a', href=True)

        # Return the link
        if link:
            return link['href']

    # If the GET request is not successful, return None
    return None

# Add a new column to the dataframe for the correct sports-reference links
df['Correct Link'] = df['Player Name'].apply(get_correct_link)

# Pause for a second between requests to avoid overwhelming the server
time.sleep(1)


    