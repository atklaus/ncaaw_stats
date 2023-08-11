import requests
from lxml import html
import pandas as pd
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
import requests
from bs4 import BeautifulSoup
import time
import random

base = 'https://www.sports-reference.com/cbb/players/{}-1.html'

# wnba_df = pd.read_csv('all_wnba_players.csv')
wnba_df = pd.read_csv('use_data/all_wnba.csv')
wnba_df.sort_values(by=['player_name'],inplace=True)
wnba_df.reset_index(inplace=True,drop=True)
# wnba_df[wnba_df['player_name']=='Kaela Davis']


def get_player_url(row):
    # Constructing the Google search URL
    player = row['player_name']
    college = row['college_team']

    user_agent = random.choice(utils.user_agents) 
    headers = {'User-Agent': user_agent} 

    query = f"{row['player_name']} college stats sports-reference women's basketball {row['college_team']}"
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    link_url = None

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the first link with "sports-reference.com" in its href attribute
        link = soup.find('a', href=lambda x: x and "www.sports-reference.com/cbb/players/" in x)

        if link:
            # Extract the desired URL using regular expression
            match = re.search(r'(https://www.sports-reference.com/.*?\.html)', link['href'])
            if match:
                link_url = match.group(1)
                print(f"For {player} from {college}, stats link: {link_url}")
            else:
                print(f"No desired pattern found in link for {player} from {college}")
        else:
            print(f"No link found for {player} from {college}")
    else:
        print(f"Failed to fetch search results for {player} from {college}")
    time.sleep(5)
    return link_url
    # To avoid making too many rapid requests, sleep for a few seconds between searches


# wnba_df['name_url'] = wnba_df['name'].str.replace(' ',"-")
# wnba_df['name_url'] = wnba_df['name_url'].str.replace("'","")
# wnba_df['name_url'] = wnba_df['name_url'].str.lower()

# # Append the url base to the 'name_url' column and store the result in a new column
# wnba_df['url'] = wnba_df['name_url'].apply(lambda x: base.format(x))
# player_urls = ['https://www.sports-reference.com/cbb/players/skylar-diggins-1.html','https://www.sports-reference.com/cbb/players/avery-warley-1.html','https://www.sports-reference.com/cbb/players/nnemkadi-ogwumike-1.html']

# for player_url in list(wnba_df['url']):
for index,row in wnba_df.iterrows():

    try:
        player_url = get_player_url(row)
        session = requests.session()
        headers, proxy_rand = utils.requst_params(utils.user_agents, utils.available_proxies)
        response = session.get(player_url, headers = headers)
        # response = session.get(player_url)
        if response.status_code != 200:
            print(response.status_code)
        else:
            pass

        page_html = BeautifulSoup(response.text, 'html5lib')
        awards,name,position,height = utils.extract_details_from_page(page_html)

        div_class = page_html.findAll('h1')
        player_name = div_class[0].find('span').text

        prefixes = {'adv_': 'Advanced', 'pg_': 'Per Game', 'tot_': 'Totals'}
        # Initialize an empty dictionary to hold dataframes
        dataframes = {}

        soup = BeautifulSoup(response.content, 'lxml')
        
        h2_tag = soup.find('h2', string='Advanced')
        table = h2_tag.find_next('table')            
        player_adv_df = pd.read_html(str(table))[0]
        dataframes['adv_'] = player_adv_df.add_prefix('adv_')

        h2_tag = soup.find('h2', string='Per Game')
        table = h2_tag.find_next('table')            
        player_pg_df = pd.read_html(str(table))[0]
        dataframes['pg_'] = player_pg_df.add_prefix('pg_')

        h2_tag = soup.find('h2', string='Totals')
        table = h2_tag.find_next('table')            
        player_tot_df = pd.read_html(str(table))[0]
        dataframes['tot_'] = player_tot_df.add_prefix('tot_')

        # Perform merging
        base_df = dataframes['pg_'].merge(dataframes['adv_'], how='left', left_on='pg_Season', right_on='adv_Season')
        base_df = base_df.merge(dataframes['tot_'], how='left', left_on='pg_Season', right_on='tot_Season')
        base_df['player_name'] =row['player_name']
        base_df['position'] =position
        base_df['height'] =height
        base_df['awards'] =awards
        base_df.to_csv('ncaa_ref/' + player_name + '.csv')
        time.sleep(10)

    except Exception as error:
    # handle the exception
        print(row['player_name']+ "ERROR:", error)


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
