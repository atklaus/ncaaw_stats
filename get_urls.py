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
import json

os.chdir('/Users/adamklaus/Documents/Personal/Develop/ncaaw_stats')
import constants as cs
import utils

importlib.reload(utils)
importlib.reload(cs)

web_dict = utils.read_json('web_dict.json')
s=utils.login()


class HerHoops:
    """
    The primary focus of this class is to create a dictionary of dictionaries contains all of the necessary urls
    to pull any team/player data from a given page
    """
    def __init__(self):
        self._s = utils.login()
        self.get_teams_url()
        self.create_master_dict()
        self.get_all_teams_seasons_url()

    def get_teams_url(self):
        """
        get urls for all teams from the herhoopsstats rankings page
        """
        page_html = utils.get_html(self._s, cs.TEAMS_URL)
        team_df = pd.read_html(str(page_html))[0] #Find way to determine this without index
        href_dict = get_url_dict(page_html)
        team_df['url'] = team_df['Team'].map(href_dict)
        self.teams_dict = dict(zip(team_df['Team'],team_df['url']))

    def read_url_dict(self):


    def create_url_dict(self):
        """
        create a dictionary of all urls
        """
        web_dict = {}
        web_dict['Teams'] = {}
        for team in teams_dict.keys():
            # print(team)
            web_dict['Teams'][team] = {'Home':teams_dict[team]}
        self.web_dict = web_dict

    def get_team_season_url(self, find_team):
        """
        create a dictionary for a given team for a given season
        """
        web_dict = self.web_dict
        team_url = cs.HOME_URL + web_dict['Teams'][team]['Home']
        page_html = utils.get_html(self._s, team_url)
        url_dict = utils.get_url_dict(page_html)
        team_page_df = pd.read_html(str(page_html))[0]
        team_page_df['url'] = team_page_df['season'].map(url_dict)
        season_dict = dict(zip(team_page_df['season'],team_page_df['url']))
        return season_dict

    def get_all_teams_seasons_url(self):
        """
        add a link to each season from each team
        """
        for team in web_dict['Teams'].keys():
            season_dict = self.get_team_season_url(team)
            for season in season_dict.keys():
                web_dict['Teams'][team][season] = season_dict[season]

        self.web_dict = web_dict

    def get_player_url(self, find_player, player_team):
        s = self.s
        web_dict = self.web_dict

        for season in web_dict['Teams'][player_team].keys():
            if season != 'Home':
                season_url = cs.HOME_URL + web_dict['Teams'][team]
                page_html = utils.get_html(s, season_url)
                url_dict = utils.get_url_dict(page_html)

        return url_dict[find_player]

    
    def get_all_player_url(self):
        """
        will need to check if player url is already in dict
        """
        web_dict = self.web_dict
        players_list = None

        web_dict['Players'] = {}
        for team in list(web_dict['Teams'].keys()):
            try:
                for season in web_dict['Teams'][team].keys():
                    if season != 'Home':
                        season_url = cs.HOME_URL + web_dict['Teams'][team][season]
                        page_html = utils.get_html(s, season_url)
                        url_dict = utils.get_url_dict(page_html)
                        season_df = utils.get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')
                        season_df['url'] = season_df['Player'].map(url_dict)
                        player_dict = dict(zip(season_df['Player'],season_df['url']))

                        for player in player_dict.keys():
                            if player not in web_dict['Players'].items():
                                # web_dict['P'][team] = {'Home':teams_dict[team]}
                                web_dict['Players'].update({player: player_dict[player]})
            except:
                print('WARNING: ' + team + ' had error')

        self.web_dict = web_dict



# utils.write_dict_to_json(web_dict,'final_web_dict.json')




for i, team in enumerate(list(web_dict['Teams'].keys())):
    if team == 'Northwestern St.':
        print(i)


season = list(season_dict.keys())[0]

urlObj = HerHoops()
teams_dict = urlObj.teams_dict
teams_dict.keys()
team = list(teams_dict.keys())[89]

web_dict = urlObj.web_dict
find_team = 'Oregon'

#Create a class, when instantiated, creates a giant dictionary of all the links needed to get to any part of the website

#Teams url


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
player_url = cs.HOME_URL + web_dict['Players'][find_player]

response = s.get(player_url, headers = dict(referer = player_url))
page_html = BeautifulSoup(response.text, 'html5lib')
url_dict = utils.get_url_dict(page_html)
player_df = utils.get_table_by_elm_text(page_html, 'Per Game', 'h2', 'card mb-3')
player_totals_df = utils.get_table_by_elm_text(page_html, 'Totals', 'h2', 'card mb-3')
player_adv_df = utils.get_table_by_elm_text(page_html, 'Advanced', 'h2', 'card mb-3')
player_val_df = utils.get_table_by_elm_text(page_html, 'Value', 'h2', 'card mb-3')
player_conf_df = utils.get_table_by_elm_text(page_html, 'Conference Per Game', 'h2', 'card mb-3')
player_totals_conf_df = utils.get_table_by_elm_text(page_html, 'Conference Totals', 'h2', 'card mb-3')
player_adv_conf_df = utils.get_table_by_elm_text(page_html, 'Conference Advanced', 'h2', 'card mb-3')
player_val_conf_df = utils.get_table_by_elm_text(page_html, 'Conference Value', 'h2', 'card mb-3')



#MAKE A LIST OF COLUMNS THAT NEED TO BE EXPANDED
# split_df = player_df['G'].str.split(expand=True)
# split_df.columns = ['G_' + x for x in ['total','perc','rank']] 







