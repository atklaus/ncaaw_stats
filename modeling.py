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
from sklearn.metrics import confusion_matrix, accuracy_score


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

####################################
# EDA
####################################

# A bar plot for the top 10 college teams with the most entries
plt.figure(figsize=(10,6))
model_df['college_team'].value_counts()[:10].plot(kind='bar')
plt.title('Top 10 College Teams')
plt.xlabel('College Team')
plt.ylabel('Count')
plt.show()

####################################
# Feature Engineering
####################################

# 1. Convert categorical variables into numerical
# We'll use LabelEncoder for this example, but depending on the situation, one-hot encoding might be more appropriate
model_df_copy = model_df.copy()

categorical_cols = ['conference','college_team','position']

model_df = pd.get_dummies(model_df_copy, prefix='conference', columns=categorical_cols)

counts = model_df_copy['college_team'].value_counts()
teams_more_than_20 = counts[counts > 15].index
college_team_dummies = pd.get_dummies(model_df_copy['college_team'], prefix='college_team')
college_team_dummies = college_team_dummies.loc[:, college_team_dummies.columns.str.startswith('college_team_') & college_team_dummies.columns.str.endswith(tuple(teams_more_than_20))]
model_df_copy = pd.concat([model_df_copy, college_team_dummies], axis=1)

counts = model_df_copy['conference'].value_counts()
teams_more_than_20 = counts[counts > 20].index
conf_dummies = pd.get_dummies(model_df_copy['conference'], prefix='conference')
conf_dummies = conf_dummies.loc[:, conf_dummies.columns.str.startswith('conference_') & conf_dummies.columns.str.endswith(tuple(teams_more_than_20))]
model_df_copy = pd.concat([model_df_copy, conf_dummies], axis=1)
model_df_copy = model_df_copy.replace({True: 1, False: 0})
model_df = model_df_copy.copy()

model_df.drop(columns=['college_team','conference','player_name','adv_season','position'],inplace=True,axis=0)


def count_award_occurrences(awards_list, award):
    count = 0
    for item in awards_list:
        if award in item:
            # Extract the number of times the award was received
            times_received = item.split('x')[0].strip()
            if times_received.isdigit():
                count += int(times_received)
            else:
                # If there's no 'x' prefix, the award was received once
                count += 1
    return count

# Apply the function to create the count columns for each award
model_df['All_Freshman_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'All-Freshman') if isinstance(x, str) else 0)
model_df['POY_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'POY') if isinstance(x, str) else 0)
model_df['NCAA_Champion_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'NCAA Champion') if isinstance(x, str) else 0)
model_df['NCAA_All_Tourney_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'NCAA All-Tourney') if isinstance(x, str) else 0)
model_df['NCAA_All_Region_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'NCAA All-Region') if isinstance(x, str) else 0)
model_df['Naismith_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'Naismith') if isinstance(x, str) else 0)
model_df['AP_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'AP') if isinstance(x, str) else 0)
model_df['ROY_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'ROY') if isinstance(x, str) else 0)
model_df['DPOY_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'DPOY') if isinstance(x, str) else 0)
model_df['All_Defense_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'All-Defense') if isinstance(x, str) else 0)
model_df['MOP_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'MOP') if isinstance(x, str) else 0)
model_df['MIP_count'] = model_df['awards'].apply(lambda x: count_award_occurrences(x.split(','), 'MIP') if isinstance(x, str) else 0)

model_df.drop(columns=['awards'],inplace=True,axis=0)

model_df.columns
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
imputer = SimpleImputer(strategy='median')
# perform the imputation
imputed_values = imputer.fit_transform(model_df[numerical_cols])
model_df[numerical_cols] = pd.DataFrame(imputed_values, columns=numerical_cols)

####################################
# EDA
####################################

# Assuming your dataframe is named df
# 1. Descriptive statistics
print(model_df.describe())

# A bar plot for the top 10 college teams with the most entries
plt.figure(figsize=(10,6))
model_df['All_Freshman_count'].value_counts()[:10].plot(kind='bar')
plt.title('Made Conference All Freshman Team')
plt.xlabel('All Freshman Team')
plt.ylabel('Count')
plt.show()

# A histogram of the 'pg_pts' column
plt.figure(figsize=(10,6))
sns.histplot(model_df['adv_ws_48_pro'], kde=True)
plt.title('Distribution of Win Shares')
plt.xlabel('Win Shares')
plt.ylabel('Frequency')
plt.xlim([-1, 1])  # adjust the range here
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(model_df['height'], kde=True)
plt.title('Distribution of Height (cm)')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.xlim([150, 210])  # adjust the range here
plt.show()


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
features = model_df.drop(columns=['adv_per_pro','adv_ws_48_pro',], axis=1)
# target = model_df['adv_ws_48_pro']
target = model_df['adv_ws_48_pro'].apply(lambda x: 1 if x > 0 else 0)

# Identify numerical columns which needs to be normalized
norm_cols = [col for col in features.columns if col not in non_norm_cols]

# Create the scaler object
scaler = MinMaxScaler()

# Fit and transform the numerical features
features[norm_cols] = scaler.fit_transform(features[norm_cols])

####################################
# Feature Selection
####################################

corr_matrix = features.corr().abs()

# Select the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

# Find index of feature columns with correlation greater than 0.85
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

# Drop highly correlated features
X = features.drop(to_drop, axis=1)
y = target.copy()

# Run a random forest to check feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X,y)
feature_importances = pd.DataFrame(rf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)

n = len(feature_importances)
importances = rf.feature_importances_
indices = np.argsort(importances)[-n:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Select only top 17 features based on importance
X = X[feature_importances.nlargest(20, 'importance').index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####################################
# Random Forest
####################################

from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the parameters for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Use the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Top n feature importances
n = len(best_model.feature_importances_)
importances = best_model.feature_importances_
indices = np.argsort(importances)[-n:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

####################################
# SVM
####################################

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Set up parameter grid for SVM
param_grid_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}

# Create SVM grid search classifier
grid_svm = GridSearchCV(svm.SVC(), param_grid_svm, refit = True, verbose = 3)

# Fit the model
grid_svm.fit(X_train, y_train)

# Print best parameters after tuning
print(grid_svm.best_params_)

# Predict
grid_predictions_svm = grid_svm.predict(X_test)

# Evaluation
print("Accuracy SVM:", metrics.accuracy_score(y_test, grid_predictions_svm))
print(classification_report(y_test, y_pred))

####################################
# XGBOOST
####################################

from xgboost import XGBClassifier

# Set up parameter grid for XGBoost
param_grid_xgb = {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5], 'max_depth': [3, 5, 7, 9]}

# Create XGBoost grid search classifier
grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss"), param_grid_xgb, refit = True, verbose = 3)

# Fit the model
grid_xgb.fit(X_train, y_train)

# Print best parameters after tuning
print(grid_xgb.best_params_)

# Predict
y_pred = grid_xgb.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Evaluation
# print("Accuracy XGBoost:", metrics.accuracy_score(y_test, grid_predictions_xgb))
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute the probabilities of the positive class
y_pred_prob = grid_xgb.predict_proba(X_test)[:, 1]

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# Compute the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

####################################
# REGRESSION: Build Model
####################################

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200], # Number of trees in random forest
    'max_features': ['auto', 'sqrt'], # Number of features to consider at every split
    'max_depth': [10, 20, 30, None], # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required at each leaf node
    'bootstrap': [True, False] # Method of selecting samples for training each tree
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Train the model with the best parameters
best_rf = grid_search.best_estimator_

# Get the feature importances
importances = best_rf.feature_importances_

# Create a DataFrame for visualization
feature_list = list(features.columns)
feature_importances = pd.DataFrame({'feature': feature_list, 'importance': importances})

# Sort the DataFrame by importance in descending order
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the top n features
n = 10  # or replace with any number you want
print(feature_importances.head(n))

####################################
# Model Evaluation
####################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the forest's predict method on the test data
predictions = best_rf.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', round(mae, 2))

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', round(mse, 2))

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print('R-squared Score:', round(r2, 2))

# Print out the mean absolute error (mae)
print('Average model error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


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
