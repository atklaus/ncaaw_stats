import os
import pandas as pd
import numpy as np

# Function to filter values
def filter_func(x, year):
    try:
        return int(x) > year
    except ValueError:
        return x == 'Career'


def convert_columns_to_lowercase(df):
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Remove columns with 'unnamed' in their names
    df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

    return df

def fill_pg_conf(df):
    # Convert column names to lowercase
    df = convert_columns_to_lowercase(df)

    # Group by 'name' and 'pg_season' and aggregate 'pg_conf'
    grouped_df = df.groupby(['name', 'pg_season'])['pg_conf'].last().reset_index()

    # Pivot the aggregated DataFrame
    pivot_table = grouped_df.pivot(index='name', columns='pg_season', values='pg_conf')

    # Fill NaN values with the last non-null value in each row
    pivot_table = pivot_table.ffill(axis=1)

    # Unpivot the pivot table to convert it back to the original format
    df = pivot_table.unstack().reset_index(name='pg_conf')
    return df



##########################################
# BUILD NCAA
##########################################

folder_path = 'ncaa_ref/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dfs = []

# Collect all unique columns from all CSVs
all_columns = set()

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    all_columns.update(df.columns)

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    name = csv_file.replace('.csv','')
    df = pd.read_csv(file_path)
    
    # Add missing columns with NaN values
    for col in all_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    df['name'] = name
    dfs.append(df)

# for csv_file in csv_files:
#     file_path = os.path.join(folder_path, csv_file)
#     name = csv_file.replace('.csv','')
#     df = pd.read_csv(file_path)
#     df['name'] = name

#     dfs.append(df)

ncaa_df = pd.concat(dfs, axis=0, ignore_index=True)
# ncaa_df['name'] = ncaa_df['name'].str.lower()
# ncaa_df[ncaa_df['name']=='kim smith']
cleaned_df = fill_pg_conf(ncaa_df)
cleaned_df = cleaned_df[cleaned_df['pg_season'] == 'Career']
cleaned_df['conference'] = cleaned_df['pg_conf']
ncaa_df = ncaa_df.merge(cleaned_df[['name','conference']], on='name', how='left')
ncaa_df= convert_columns_to_lowercase(ncaa_df)
ncaa_df = ncaa_df.loc[ncaa_df['pg_season'].apply(lambda x: filter_func(str(x),1997))]
ncaa_df = ncaa_df[ncaa_df['pg_season'] == 'Career']
ncaa_df.dropna(subset=['tot_pts'],inplace=True)
# cleaned_df[cleaned_df['name'] =='Tamika Whitmore'].to_excel('test.xlsx')
# ncaa_df.to_excel('use_data/all_ncaa_updated.xlsx')
ncaa_df.to_csv('use_data/all_ncaa_updated.csv')

##########################################
# BUILD WNBA
##########################################

def add_first_year_column(df, column_name="first_year"):
    """
    Add a column to the dataframe indicating each player's first year in the WNBA.
    
    Args:
    - df (pd.DataFrame): The dataset containing player data.
    - column_name (str): The name of the column to be added for the first year.
    
    Returns:
    - pd.DataFrame: The dataset with the added 'first_year' column.
    """
    # Exclude rows with 'Career' as they don't represent individual years
    df_filtered = df.dropna(subset=['pg_year_num'])
    
    # Find the first year for each player
    first_year_series = df_filtered.groupby('player_name')['pg_year'].min()
    
    # Map the first year to the original dataframe
    df[column_name] = df['player_name'].map(first_year_series)
    
    return df

folder_path = 'wnba_ref/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dfs = []

# for csv_file in csv_files:
#     file_path = os.path.join(folder_path, csv_file)
#     name = csv_file.replace('.csv','')
#     df = pd.read_csv(file_path)
#     df['name'] = name

#     dfs.append(df)

# Collect all unique columns from all CSVs
all_columns = set()

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    all_columns.update(df.columns)

# Convert the set to a list to ensure a consistent order
all_columns = list(all_columns)

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    name = csv_file.replace('.csv','')
    df = pd.read_csv(file_path)
    
    # Add the 'name' column before reordering
    df['name'] = name
    
    # Add missing columns with NaN values
    for col in all_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Reorder columns to match the 'all_columns' list
    df = df[all_columns + ['name']]
    
    dfs.append(df)

# Concatenate all DataFrames
wnba_df = pd.concat(dfs, axis=0, ignore_index=True)
wnba_df=convert_columns_to_lowercase(wnba_df)
# wnba_df[wnba_df['player_name'] =='Nneka Ogwumike'].to_excel('test.xlsx')

# Convert the 'pg_year' column to numeric and handle errors by setting non-convertible values to NaN
wnba_df['pg_year_num'] = pd.to_numeric(wnba_df['pg_year'], errors='coerce')
wnba_df = add_first_year_column(wnba_df, column_name="debut_year")
wnba_df = wnba_df.loc[wnba_df['pg_year'].apply(lambda x: filter_func(str(x),year=2002))]
wnba_df = wnba_df.dropna(subset=['college_team'])
wnba_df = wnba_df[(wnba_df['pg_year'] == 'Career')][['player_name','college_team','adv_ws/48','adv_per','debut_year']]
wnba_df.to_csv('use_data/all_wnba.csv')



# ########################################################################################################

# import pandas as pd
# from fuzzywuzzy import process
# from fuzzywuzzy import fuzz

# def match_name(name, list_names, min_score=0):
#     # -1 score incase we don't get any matches
#     max_score = -1
#     # Returning empty name for no match as well
#     max_name = ""
#     # Iternating over all names in the other
#     for name2 in list_names:
#         #Finding fuzzy match score
#         score = process.extractOne(name, [name2],scorer=fuzz.token_set_ratio)[1]
#         # Checking if we are above our threshold and have a better score
#         if (score > min_score) & (score > max_score):
#             max_name = name2
#             max_score = score
#     return max_name, max_score

# # List of names
# names_df1 = filtered_df['name'].tolist()
# names_df2 = ncaa_df['name'].tolist()

# # Dictionary of matched names
# name_match = {}

# # For each name in df1
# for name1 in names_df1:
#     # Find the best match in df2
#     match = match_name(name1, names_df2, 90)
    
#     # New dict for storing data
#     match_dict = {
#         "df1_name": name1,
#         "df2_name": match[0],
#         "score": match[1]
#     }

#     # print match_dict

#     # Append the match dict to our final list
#     name_match[name1] = match_dict

# name_match
# # Now you can use this dictionary to map names in df1 to the corresponding ones in df2
# df1["matched_name"] = df1["Name"].map(name_match)

