import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

####################################
# Clean Data
####################################

def convert_columns_to_lowercase(df):
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Remove columns with 'unnamed' in their names
    df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

    return df


ncaa_df = pd.read_csv('use_data/all_ncaa.csv')
wnba_df = pd.read_csv('use_data/all_wnba.csv')
use_df = ncaa_df.merge(wnba_df, left_on='name', right_on='player_name', how='left',)
use_df.rename(columns={'adv_per_x': 'adv_per_college','adv_per_y':'adv_per_pro','player_name_x':'player_name'}, inplace=True)
use_df = convert_columns_to_lowercase(use_df)
use_df.drop(columns =['player_name_y','name','pg_school','pg_season','adv_class', 'pg_class', 'adv_school', 'pg_conf'], axis=1, inplace=True)
use_df.reset_index(drop=True, inplace=True)
use_df = use_df.dropna(how='all', axis=1)


####################################
# EDA
####################################

# A bar plot for the top 10 college teams with the most entries
plt.figure(figsize=(10,6))
use_df['college_team'].value_counts()[:10].plot(kind='bar')
plt.title('Top 10 College Teams')
plt.xlabel('College Team')
plt.ylabel('Count')
plt.show()

####################################
# Feature Engineering
####################################

# 1. Convert categorical variables into numerical
# We'll use LabelEncoder for this example, but depending on the situation, one-hot encoding might be more appropriate
use_df_copy = use_df.copy()

categorical_cols = ['conference','college_team']

use_df = pd.get_dummies(use_df_copy, prefix='conference', columns=['conference','college_team'])

counts = use_df_copy['college_team'].value_counts()
teams_more_than_20 = counts[counts > 15].index
college_team_dummies = pd.get_dummies(use_df_copy['college_team'], prefix='college_team')
college_team_dummies = college_team_dummies.loc[:, college_team_dummies.columns.str.startswith('college_team_') & college_team_dummies.columns.str.endswith(tuple(teams_more_than_20))]
use_df_copy = pd.concat([use_df_copy, college_team_dummies], axis=1)

counts = use_df_copy['conference'].value_counts()
teams_more_than_20 = counts[counts > 20].index
conf_dummies = pd.get_dummies(use_df_copy['conference'], prefix='conference')
conf_dummies = conf_dummies.loc[:, conf_dummies.columns.str.startswith('conference_') & conf_dummies.columns.str.endswith(tuple(teams_more_than_20))]
use_df_copy = pd.concat([use_df_copy, conf_dummies], axis=1)
use_df_copy = use_df_copy.replace({True: 1, False: 0})
model_df = use_df_copy.copy()

model_df.drop(columns=['college_team','conference','player_name','adv_season'],inplace=True,axis=0)

all_columns = model_df.columns
non_categorical_cols = [col for col in all_columns if col not in categorical_cols]

for col in non_categorical_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
numerical_cols = model_df.select_dtypes(include=['float64', 'int64']).columns

model_df.to_excel('model_df.xlsx')

####################################
# Impute Missing Values
####################################

# Create an imputer object
imputer = SimpleImputer(strategy='mean')
# perform the imputation
imputed_values = imputer.fit_transform(model_df[numerical_cols])
model_df[numerical_cols] = pd.DataFrame(imputed_values, columns=numerical_cols)


####################################
# EDA
####################################

# Assuming your dataframe is named df
# 1. Descriptive statistics
print(model_df.describe())

# A histogram of the 'pg_pts' column
plt.figure(figsize=(10,6))
sns.histplot(model_df['pg_pts'], kde=True)
plt.title('Distribution of Points Per Game')
plt.xlabel('Points Per Game')
plt.ylabel('Frequency')
plt.xlim([0, 30])  # adjust the range here
plt.show()

# A boxplot of 'pg_pts' grouped by 'conference'
plt.figure(figsize=(10,6))
sns.boxplot(x='conference', y='pg_pts', data=model_df)
plt.title('Points Per Game Distribution by Conference')
plt.xticks(rotation=90)  # rotate x-axis labels
plt.ylim([0, 30])  # adjust the range here
plt.show()


# Correlation matrix
corr_mat = model_df.corr()

# Heatmap of correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(corr_mat, annot=True)
plt.title('Correlation matrix of variables')
plt.show()

plt.scatter(model_df['pg_g'], model_df['pg_gs'])
plt.xlabel('pg_g')
plt.ylabel('pg_gs')
plt.title('pg_g vs pg_gs')
plt.show()

sns.pairplot(model_df)
plt.show()

####################################
# Normalize Data
####################################

# List comprehension to get the column names with 'college_team' prefix
college_team_cols = [col for col in model_df.columns if col.startswith('college_team')]

# List comprehension to get the column names with 'conference' prefix
conference_cols = [col for col in model_df.columns if col.startswith('conference')]
non_norm_cols = college_team_cols + conference_cols

# Separate features and target
features = model_df.drop(columns=['adv_per_pro'], axis=1)
target = model_df['adv_per_pro']

# Identify numerical columns which needs to be normalized
norm_cols = [col for col in features.columns if col not in non_norm_cols]

# Create the scaler object
scaler = MinMaxScaler()

# Fit and transform the numerical features
features[norm_cols] = scaler.fit_transform(features[norm_cols])

####################################
# Build Model
####################################

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(features_train, target_train)

# Get the feature importances
importances = rf.feature_importances_

# Create a DataFrame for visualization
feature_list = list(features.columns)
feature_importances = pd.DataFrame({'feature': feature_list, 'importance': importances})

# Sort the DataFrame by importance in descending order
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the top n features
n = 10  # or replace with any number you want
print(feature_importances.head(n))

####################################
# Denormalize Data
####################################

# Assuming 'predictions' is the output from your model
predictions = scaler.inverse_transform(predictions)

# Print the denormalized predictions
print(predictions)

# Convert ndarray to DataFrame
predictions = pd.DataFrame(predictions, columns=norm_cols)

# Print the denormalized predictions
print(predictions)
