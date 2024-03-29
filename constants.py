import os
import json

# Relative paths for configuring on different laptops
base_path = os.path.abspath(os.path.dirname(__file__))
# pull_data_path = os.path.join(base_path, 'pull_data')

os.chdir(base_path)

LOGIN_URL = 'https://herhoopstats.com/accounts/login/?return_url=/'
HOME_URL = 'https://herhoopstats.com'
NCAA_TEAMS_URL = 'https://herhoopstats.com/stats/ncaa/research/team_single_seasons/?min_season=2023&max_season=2023&season_type=reg&submit=true&stats_to_show=per_game&criteria1=pts_per_game&comp1=ge&threshold1=15'
TABLE_CLASS = 'card mb-3'

PRO_HOME_URL = 'https://herhoopstats.com/stats/wnba/research/team_single_seasons/?min_season=2020&max_season=2020&season_type=reg&stats_to_show=summary&submit=true'


PLAYER_INPUT_DICT = {
    'player_per_game_df':{'table':'Per Game','prefix':'per_game_'},
    'player_totals_df':{'table':'Totals','prefix':'totals_'},
    'player_adv_df':{'table':'Advanced','prefix':'adv_'},
    'player_val_df':{'table':'Value','prefix':'val_'},
    'player_conf_per_game_df':{'table':'Conference Per Game','prefix':'conf_per_game_'},
    'player_conf_totals_df':{'table':'Conference Totals','prefix':'conf_totals_'},
    'player_conf_adv_df':{'table':'Conference Advanced','prefix':'conf_adv_'},
    'player_conf_val_df':{'table':'Conference Value','prefix':'conf_val_'},

}

import json
  
# Opening JSON file
f = open('creds.json')
  
# returns JSON object as 
# a dictionary
CREDS_DICT = json.load(f)

# #natl vs conf
# with open("creds.txt") as f:
#     for line in f:
#         CREDS_DICT = json.loads(line)



