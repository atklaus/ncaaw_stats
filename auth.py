import requests
from lxml import html
import os
import pandas as pd
import html5lib
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import requests
import lxml.html as lh
import importlib


os.chdir('/Users/adamklaus/Documents/Personal/Develop/herhoopstats')

import constants as cs


### Here, we're getting the login page and then grabbing hidden form
### fields.  We're probably also getting several session cookies too.

def login():
    s = requests.session()
    login = s.get(cs.LOGIN_URL)
    login_html = lxml.html.fromstring(login.text)
    hidden_inputs = login_html.xpath(r'//form//input[@type="hidden"]')
    form = {x.attrib["name"]: x.attrib["value"] for x in hidden_inputs}
    print(form)
    # {'csrftok': '9e34ca7e492a0dda743369433e78ccf10c1e68bbb1f453cbb80ce6eaeeebe928', 
    #  'context': ''}
    
    form['email'] = 'atklaus@wisc.edu'
    form['password'] = 'Cubswin(11)!'
    response = s.post(cs.LOGIN_URL, data=form, headers = dict(referer=cs.LOGIN_URL))
    return s


def get_table_by_elm_text(page_html, find_text, element, class_name):

    table_df = None
    div_class = page_html.findAll('div',class_=class_name)

    for div in div_class:
        try:
            if find_text in div.find(element).text:
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


def add_url_col(df, href_dict, col):
    for index, key in enumerate(df[col]):
        df.loc[index, 'url'] = href_dict[key] 


    return df


s = login()

# url = 'https://herhoopstats.com/stats/ncaa/team/2021/natl/northwestern-wildcats-womens-basketball-stats-11e8e149-c9f6-64de-af82-12df17ae4e1e/'

importlib.reload(cs)

soup = BeautifulSoup(response.text, 'lxml')
schedule = page_html.find_all(id='schedule')

#Teams url
response = s.get(cs.TEAMS_URL, headers = dict(referer = cs.TEAMS_URL))
page_html = BeautifulSoup(response.text, 'html5lib')
team_df = pd.read_html(str(page_html))[0] #Find way to determine this without index
href_dict = get_url_dict(page_html)
team_df['url'] = team_df['Team'].map(href_dict)

#Team url
find_team = 'Oregon'
team_url = 'https://herhoopstats.com/' + team_df[team_df['Team'] == find_team]['url'].iloc[0]
response = s.get(team_url, headers = dict(referer = team_url))
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = get_url_dict(page_html)
team_page_df = pd.read_html(str(page_html))[0]
team_page_df['url'] = team_page_df['season'].map(url_dict)

#Season url
find_season = '2019-20'
season_dict = dict(zip(team_page_df['season'], team_page_df['url']))
season_url = cs.HOME_URL + season_dict[find_season]
response = s.get(season_url, headers = dict(referer = season_url))
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = get_url_dict(page_html)
player_df = get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')
player_df['url'] = player_df['Player'].map(url_dict)


#player url
find_player = 'Sabrina Ionescu'
player_url = cs.HOME_URL + url_dict[find_player]
response = s.get(player_url, headers = dict(referer = player_url))
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = get_url_dict(page_html)
player_df = get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')



player_totals_df = get_table_by_elm_text(page_html, 'Roster Totals', 'h2', 'card mb-3')


# player = page_html.find_all(id='player_stats')


pd.read_html(str(player))

player
soup = BeautifulSoup(player, 'html5lib')

test = soup.find("h2",attrs={ "class" : "color_3"})

soup.select(str(test))



text = 'Roster Per Game'
element = 'h2'
class_name = 'card mb-3'


get_table_by_elm_text(page_html, text, element, class_name)



table_df[0]


