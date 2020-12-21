import os
import json

# Relative paths for configuring on different laptops
base_path = os.path.abspath(os.path.dirname(__file__))
# pull_data_path = os.path.join(base_path, 'pull_data')

os.chdir(base_path)

LOGIN_URL = 'https://herhoopstats.com/accounts/login/?return_url=/'
HOME_URL = 'https://herhoopstats.com/'
TEAMS_URL = 'https://herhoopstats.com/stats/leaderboard/'
TABLE_CLASS = 'card mb-3'

#natl vs conf

with open("creds.txt") as f:
    for line in f:
        CREDS_DICT = json.loads(line)



