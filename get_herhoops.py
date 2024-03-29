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
        # self.read_url_dict()

    def get_teams_url(self):
        """
        get urls for all teams from the herhoopsstats rankings page
        """
        # NCAA_TEAMS_URL = 'https://herhoopstats.com/stats/ncaa/research/team_total_seasons/?division=1&min_season=2010&max_season=2023&games=all&submit=true'
        NCAA_TEAMS_URL = 'https://herhoopstats.com/stats/ncaa/research/team_total_games/?division=1&min_season=2010&max_season=2023&result=both&loc_h=1&loc_a=1&loc_n=1&submit=true'

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
        for team in self.teams_dict.keys():
            # print(team)
            web_dict['Teams'][team] = {'Home':self.teams_dict[team]}
        self.web_dict = web_dict

    def get_team_season_url(self, find_team):
        """
        create a dictionary for a given team for a given season
        """
        web_dict = self.web_dict
        team_url = HOME_URL + web_dict['Teams'][find_team]['Home']
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
        for team in self.web_dict['Teams'].keys():
            season_dict = self.get_team_season_url(team)
            for season in season_dict.keys():
                self.web_dict['Teams'][team][season] = season_dict[season]

        self.web_dict = self.web_dict

    def get_player_url(self, find_player, player_team):
        s = self._s
        web_dict = self.web_dict

        for season in web_dict['Teams'][player_team].keys():
            if season != 'Home':
                season_url = HOME_URL + web_dict['Teams'][player_team]
                page_html = utils.get_html(s, season_url)
                url_dict = utils.get_url_dict(page_html)

        return url_dict[find_player]

    
    def get_all_player_url(self):
        """
        will need to check if player url is already in dict
        """
        web_dict = self.web_dict

        web_dict['Players'] = {}
        for team in list(web_dict['Teams'].keys()):
            try:
                for season in web_dict['Teams'][team].keys():
                    if season != 'Home':
                        season_url = HOME_URL + web_dict['Teams'][team][season]
                        page_html = utils.get_html(self.s, season_url)
                        url_dict = utils.get_url_dict(page_html)
                        season_df = utils.get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')
                        season_df['url'] = season_df['Player'].map(url_dict)
                        player_dict = dict(zip(season_df['Player'],season_df['url']))

                        for player in player_dict.keys():
                            if player not in web_dict['Players'].items():
                                web_dict['P'][team] = {'Home':self.teams_dict[team]}
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



# NCAA_TEAMS_URL = 'https://herhoopstats.com/stats/ncaa/research/team_single_seasons/?division=1&min_season=2023&max_season=2023&games=all&criteria0=pts_per_game&comp0=ge&threshold0=15&stats_to_show=summary&submit=true'
statsObj = HerHoopsData()
statsObj.get_teams_url()
statsObj.create_url_dict()

team = 'Iowa Hawkeyes'

def get_tables_list(page_html,tables_list,join_key='season',table_num=0):
    keep_same = {'season','Player'}
    tables = pd.read_html(str(page_html))
    base_df = tables[table_num]

    for table in tables_list:
        try:
            new_df = utils.get_table_by_elm_text(page_html, table, 'h2', 'card mb-3')
            suffix = '_'+ table.replace(' ','_').lower()
            new_df.columns = ['{}{}'.format(c, '' if c in keep_same else suffix)for c in new_df.columns]
            base_df = base_df.merge(new_df,on=join_key,how='left')
        except:
            pass
    return base_df



def read_csvs_from_folder(folder_path):

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    final_df = pd.concat(dfs, axis=0, ignore_index=True)

    for col in final_df.columns:
        if '  ' in str(final_df[col][0]):
            final_df[col] = final_df[col].str.split('  ').str[0]
            
        if pd.api.types.is_numeric_dtype(final_df[col]):
            final_df[col] = final_df[col].astype(float)
    
    return final_df


teams_list = list(statsObj.web_dict['Teams'].keys())
teams_list.sort()

for team in teams_list:
    team_url = HOME_URL + statsObj.web_dict['Teams'][team]['Home']
    page_html = utils.get_html(statsObj._s, team_url)
    url_dict = utils.get_url_dict(page_html)
    tables_list = ['Season Summary Advanced','Per Game','Advanced','Conference Summary','NCAA Tournament Summary']
    team_page_df = get_tables_list(page_html,tables_list,join_key='season')
    team_page_df['url'] = team_page_df['season'].map(url_dict)
    team_page_df['team_name'] = team
    team_page_df.to_csv('teams/' + team + '.csv')

    #Get all seasons per team
    season_dict = dict(zip(team_page_df['season'],team_page_df['url']))
    season_dict = statsObj.get_team_season_url(team)

    statsObj.web_dict['Teams'][team]['seasons'] = season_dict

    #Get all players for a team


    # for season in season_dict.keys():
    #     statsObj.web_dict['Teams'][team][season] = season_dict[season]

teams_df = read_csvs_from_folder(folder_path='teams/')
teams_df = teams_df.add_prefix('team_')
teams_df.sort_values(['team_team_name', 'team_season'], ascending=[True, False],inplace=True)
teams_df.reset_index(drop=True,inplace=True)
teams_df.to_csv('all_teams.csv')


statsObj.web_dict['Players'] = {}
# for team in list(statsObj.web_dict['Teams'].keys()):
for index, row in teams_df.iterrows():
    try:
        season_url = HOME_URL + row['team_url']
        page_html = utils.get_html(statsObj._s, season_url)
        url_dict = utils.get_url_dict(page_html)
        tables_list = ['Roster Totals','Roster Advanced','Roster Value']
        season_df = get_tables_list(page_html,tables_list,join_key='Player',table_num=12)
        season_df['team_name'] = row['team_team_name']
        season_df.to_csv('seasons/' + row['team_team_name'] + ' ' + row['team_season'] + '.csv')

        # season_df = utils.get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')
        # season_df['url'] = season_df['Player'].map(url_dict)
        # season_df['url'] = HOME_URL + season_df['url']
        # player_dict = dict(zip(season_df['Player'],season_df['url']))

        # for player in player_dict.keys():
        #     if player not in statsObj.web_dict['Players'].items():
        #         # web_dict['Players'][team] = {'Home':statsObj.teams_dict[team]}
        #         statsObj.web_dict['Players'].update({player: player_dict[player]})
    except:
        print('WARNING: ' + team + ' had error')

# PULL ALL PLAYERS
# for player in statsObj.web_dict['Players'].keys():
#     player_url = statsObj.web_dict['Players'][player]
#     page_html = utils.get_html(statsObj._s, player_url)
#     tables_list = ['Advanced','Value','Conference Per Game','Conference Advanced', 'Conference Value']
#     player_page_df = get_tables_list(page_html,tables_list,join_key='season')
#     player_page_df['player_name'] = player
#     player_page_df.to_csv('players/' + player + '.csv')

folder_path = 'seasons/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('2022-23.csv')]
dfs = []

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    season = csv_file.split(' ')[-1].replace('.csv','')
    df = pd.read_csv(file_path)
    df['season'] = season
    dfs.append(df)

import numpy as np

final_df = pd.concat(dfs, axis=0, ignore_index=True)
final_df.replace('--',np.nan, inplace=True)

col = 'Player Def Rtg_roster_value'

for col in final_df.columns:
    if '  ' in str(final_df[col][0]):
        final_df[col] = final_df[col].str.split('  ').str[0]
        
    if pd.api.types.is_numeric_dtype(final_df[col]):
        final_df[col] = final_df[col].astype(float)
    else:
        print(col)


final_df = final_df.apply(pd.to_numeric, errors='ignore')

pd.to_numeric(final_df['Player Def Rtg_roster_value'],errors='ignore')
# folder_path = 'players/'
# full_players_df = read_csvs_from_folder(folder_path='seasons/')
full_players_df = final_df.copy()

keep_cols = list(final_df.select_dtypes([np.number]).columns) + list(['Player','team_name','season'])


grouped_df = full_players_df[keep_cols].groupby(by=['Player','team_name'], as_index=False).max()
grouped_df = full_players_df.sort_values(by=['Player'], ascending=False).groupby(by='Player', as_index=False).first()
# grouped_df = grouped_df[['Player','team_name']]
grouped_df.to_csv('player_bio.csv')
full_players_df.to_csv('all_players.csv')
grouped_df.to_csv('ncaa_all_max.csv')


###########
# WNBA
###########

WNBA_PLAYERS_URL = 'https://herhoopstats.com/stats/wnba/research/player_total_seasons/?min_season=2008&max_season=2022&season_type=reg&submit=true'
page_html = utils.get_html(statsObj._s, WNBA_PLAYERS_URL)
# utils.get_table_by_elm_text(page_html, find_text)
player_df = pd.read_html(str(page_html))[0] #Find way to determine this without index
href_dict = utils.get_url_dict(page_html)
player_df['url'] = player_df['Player'].map(href_dict)
player_df['url'] = HOME_URL + player_df['url'] 

for index, row in player_df.iterrows():
    try:
        page_html = utils.get_html(statsObj._s, row['url'])
        tables_list = ['Totals','Advanced','Value']
        base_df = get_tables_list(page_html,tables_list,join_key='season',table_num=0)
        base_df['player_name'] = row['Player']
        base_df.to_csv('WNBA/' + row['Player'] + '.csv')

        # season_df = utils.get_table_by_elm_text(page_html, 'Roster Per Game', 'h2', 'card mb-3')
        # season_df['url'] = season_df['Player'].map(url_dict)
        # season_df['url'] = HOME_URL + season_df['url']
        # player_dict = dict(zip(season_df['Player'],season_df['url']))

        # for player in player_dict.keys():
        #     if player not in statsObj.web_dict['Players'].items():
        #         # web_dict['Players'][team] = {'Home':statsObj.teams_dict[team]}
        #         statsObj.web_dict['Players'].update({player: player_dict[player]})
    except:
        print('WARNING: ' + row['Player'] + ' had error')

pro_df = read_csvs_from_folder('wnba/')

remove_list = pro_df[(pro_df['season'] == '2009') | (pro_df['season'] == '2008')]['player_name'].unique()

pro_df = pro_df[~pro_df.player_name.isin(remove_list)]
pro_list = list(pro_df['player_name'].unique())
full_players_df['player_name'] = full_players_df['Player'].str.split('This HTML5 audio').str[0]
ncaa_players_list = list(full_players_df['player_name'].unique())

full_players_df[full_players_df['player_name'].isin(pro_list)]['player_name'].unique()


from fuzzywuzzy import process

def fuzzy_name_matching(list1, list2, threshold=90):
    matched_names = {}

    for name in list1:
        match, match_score = process.extractOne(name, list2)

        if match_score >= threshold:
            matched_names[name] = match
            list2.remove(match)

    return matched_names

matched_names = fuzzy_name_matching(pro_list, ncaa_players_list)
print(matched_names)

# import os
# import openai
# from io import StringIO
# openai.api_key = os.getenv('OPEN_AI_KEY')

# msg = """
# Can you add a columns to the following table with filled in information about each of these players hometown, height, weight and whether or not they are in the WNBA that you should be able to get from https://www.basketball-reference.com/ or https://www.sports-reference.com/cbb/? Can you also output as a csv?
# {}
# """

# # create a completion
# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{'role':'user',"content": msg.format(grouped_df)}])
# response_text = completion.choices[0].message.content

# msg = """
# Can you output a json object where the key is the initial college name and mascot from the list and the value has the mascot name removed? An example of a key value pair would be 'George Washington Colonials': 'George Washington'
# {}
# """

# teams_list

# # create a completion
# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{'role':'user',"content": msg.format(teams_list)}])
# response_text = completion.choices[0].message.content

# import json
# json.dumps(response_text)

# response_text = response_text.strip()
# new_team_list = response_text.split(',')
# new_team_list = [x.strip(' ') for x in new_team_list]



# # Process the response text into a pandas DataFrame
# df = pd.read_csv(StringIO(response_text), delimiter=",")
# df
# df.columns



