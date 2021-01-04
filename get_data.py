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
import time

os.chdir('/Users/adamklaus/Documents/Personal/Develop/ncaaw_stats')
from constants import LOGIN_URL, HOME_URL, PRO_HOME_URL, NCAA_TEAMS_URL, TABLE_CLASS, CREDS_DICT, PLAYER_INPUT_DICT
import utils

importlib.reload(utils)

class HerHoopsData:
    """
    The primary focus of this class is to create a dictionary of dictionaries contains all of the necessary urls
    to pull any team/player data from a given page
    """
    def __init__(self):
        self._s = utils.login()
        self.get_teams_url()
        self.read_url_dict()

    def get_teams_url(self):
        """
        get urls for all teams from the herhoopsstats rankings page
        """
        page_html = utils.get_html(self._s, NCAA_TEAMS_URL)
        # utils.get_table_by_elm_text(page_html, find_text)
        team_df = pd.read_html(str(page_html))[0] #Find way to determine this without index
        href_dict = utils.get_url_dict(page_html)
        team_df['url'] = team_df['Team'].map(href_dict)
        self.teams_dict = dict(zip(team_df['Team'],team_df['url']))

    def read_url_dict(self):
        self.players_dict = utils.read_json('ncaa_players_urls.json')
        self.teams_dict = utils.read_json('ncaa_teams_urls.json')        

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
        team_url = HOME_URL + web_dict['Teams'][team]['Home']
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
        s = self._s
        web_dict = self.web_dict

        for season in web_dict['Teams'][player_team].keys():
            if season != 'Home':
                season_url = HOME_URL + web_dict['Teams'][team]
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
                        season_url = HOME_URL + web_dict['Teams'][team][season]
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

                
    def get_all_player_tables(self, find_player):

        self.get_player_html(find_player)
        for df_name in PLAYER_INPUT_DICT.keys():
            self.get_player_table(find_player, df_name)

    def get_player_table(self, find_player, df_name, heading_tag='h2', _class_name='card mb-3'):

        try:
            df = utils.get_table_by_elm_text(self.page_html, PLAYER_INPUT_DICT[df_name]['table'], heading_tag, _class_name)
            df = self.split_stats_column(df)
        
            df = df.add_prefix(PLAYER_INPUT_DICT[df_name]['prefix'])
            setattr(self,df_name,df)

        except:
            #If player doesn't have data for this table
            print(find_player + ' does not have ' + df_name)
            pass
        

    def get_player_html(self, find_player):
        try:
            player_url = HOME_URL + self.players_dict[find_player]
            self.page_html = utils.get_html(self._s, player_url)
        except:
            print('WARNING: url issue with ' + player_url)
            response = self._s.get(player_url)
            self.page_html = BeautifulSoup(response.text, 'html5lib')


    def merge_all_player_tables(self, find_player):
        """
        """
        self.get_all_player_tables(find_player)
        dfs_list = list(PLAYER_INPUT_DICT.keys())
        dfs_list.remove('player_per_game_df')
        temp_player_df = self.player_per_game_df.copy()
        for temp_df in dfs_list:
            temp_player_df = temp_player_df.join(getattr(self, temp_df))

        temp_player_df['player'] = find_player

        self.player_df = temp_player_df

    def split_stats_column(self, df):
        """
        MAKE A LIST OF COLUMNS THAT NEED TO BE EXPANDED
        Add values to constants file
        """
        for col in df.columns:
            if col not in ['season','team']:
                split_df = df[col].str.split(expand=True)
                split_df.columns = [col.lower() + '_' + x for x in ['total','perc','rank']]
                df = df.join(split_df)
                del df[col]

        return df


statsObj = HerHoopsData()
players_list = list(statsObj.players_dict.keys())
players_list.sort()
players_list = players_list[1941:]
stats_df = pd.DataFrame()
count=0

for player in players_list:
    if count > 250:
        stats_df.to_csv('ncaa_player_stats_2.csv')
        count = 0

    count += 1

    statsObj.merge_all_player_tables(player)
    stats_df = pd.concat([stats_df, statsObj.player_df], sort=False)
    # statsObj.player_df.to_csv('test.csv')



stats_df.head()




