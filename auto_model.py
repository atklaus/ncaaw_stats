import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

ncaa_df = pd.read_csv('use_data/all_ncaa_updated.csv')
wnba_df = pd.read_csv('use_data/all_wnba.csv')
model_df = ncaa_df.merge(wnba_df, left_on='name', right_on='player_name', how='left',)
model_df.rename(columns={'adv_per_x': 'adv_per_college','adv_per_y':'adv_per_pro','adv_ws/48':'adv_ws_48_pro','player_name_x':'player_name'}, inplace=True)
model_df = convert_columns_to_lowercase(model_df)
model_df.drop(columns =['player_name_y','name','pg_school','pg_season','adv_class', 'pg_class', 'adv_school', 'pg_conf'], axis=1, inplace=True)
model_df.reset_index(drop=True, inplace=True)
model_df = model_df.dropna(how='all', axis=1)
model_df.to_csv('test.csv')


# Dropping the index column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Inspecting missing data
missing_data = data.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

# Display missing data
missing_data


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identifying numerical and categorical columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Removing the target variable from the list of numerical columns
num_cols = num_cols.drop('adv_ws_48_pro')

# Defining the data preprocessing steps
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipeline = Pipeline(steps=[
    ('imputer', num_imputer)
])

cat_pipeline = Pipeline(steps=[
    ('imputer', cat_imputer),
    ('onehot', one_hot_encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

# Applying the transformations
data_preprocessed = data.copy()
data_preprocessed[num_cols] = num_imputer.fit_transform(data[num_cols])
data_preprocessed[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Dropping rows with missing target variable
data_preprocessed.dropna(subset=['adv_ws_48_pro'], inplace=True)

# Display the first few rows of the preprocessed data
data_preprocessed.head()


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identifying numerical and categorical columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Removing the target variable from the list of numerical columns
num_cols = num_cols.drop('adv_ws_48_pro')

# Defining the data preprocessing steps
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipeline = Pipeline(steps=[
    ('imputer', num_imputer)
])

cat_pipeline = Pipeline(steps=[
    ('imputer', cat_imputer),
    ('onehot', one_hot_encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

# Applying the transformations
data_preprocessed = data.copy()
data_preprocessed[num_cols] = num_imputer.fit_transform(data[num_cols])
data_preprocessed[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Dropping rows with missing target variable
data_preprocessed.dropna(subset=['adv_ws_48_pro'], inplace=True)

# Display the first few rows of the preprocessed data
data_preprocessed.head()

# Checking the number of unique categories in each categorical column
unique_counts = {col: data_preprocessed[col].nunique() for col in cat_cols}
unique_counts

# Dropping the columns with too many unique categories
data_preprocessed.drop(columns=['player_name', 'awards', 'college_team'], inplace=True)

# Updating the list of categorical columns
cat_cols = cat_cols.drop(['player_name', 'awards', 'college_team'])

# Applying one-hot encoding
data_encoded = pd.get_dummies(data_preprocessed, columns=cat_cols)

# Display the first few rows of the encoded data
data_encoded.head()
