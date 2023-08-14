import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from xgboost import XGBClassifier

#########################
# SET RUN TYPE
#########################
n=10
tuned_summary_filename = "new_tot_model_evaluation_11_feat.xlsx"
# Load the data from the provided Excel file
df = pd.read_excel("model_df.xlsx")
# df = pd.read_csv("model_df_updated.csv")
target_variable = 'ws_48_pro'
# cols_to_keep = [col for col in df.columns if col.startswith('tot_')] + ['ws_48_pro']
# df = df[cols_to_keep]

#########################
# MODEL
#########################

# Load the data
# df = pd.read_csv("model_df_updated.csv")
target_variable = 'ws_48_pro'

# Fill NaN values only for 'ws_48_pro' with 0 and convert to binary classes
df['ws_48_pro'].fillna(0, inplace=True)
df['ws_48_pro'] = (df['ws_48_pro'] > 0).astype(int)
# balanced_data = pd.DataFrame()

# # Assuming 'year' is a column representing the year of the data
# for year in range(2002, 2023):
#     data_year = df[df['last_season'] == year]
#     majority = data_year[data_year['ws_48_pro'] == 0]
#     minority = data_year[data_year['ws_48_pro'] == 1]

#     # Downsample and upsample within this year
#     majority_downsampled = resample(majority, replace=False, n_samples=min(len(majority), 6000), random_state=42)  # Example size
#     minority_upsampled = resample(minority, replace=True, n_samples=len(majority_downsampled), random_state=42)

#     balanced_year_data = pd.concat([majority_downsampled, minority])
#     balanced_data = pd.concat([balanced_data, balanced_year_data])

# samples = 2000
# data_majority = df[df['ws_48_pro'] == 0]
# data_minority = df[df['ws_48_pro'] == 1]

# # Downsample the majority class to 2000 instances
# data_majority_downsampled = resample(data_majority, replace=False, n_samples=samples, random_state=42)

# # Upsample the minority class to 2000 instances
# data_minority_upsampled = resample(data_minority, replace=True, n_samples=samples, random_state=42)

# # Combine downsampled majority class and upsampled minority class
# balanced_data = pd.concat([data_majority_downsampled, data_minority_upsampled])

# Separate features and target variable
X = df.drop(columns=[target_variable])
y= df[target_variable]

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_resampled = X_imputed

# Apply SMOTE to balance the classes
# smote = SMOTE(sampling_strategy={1: samples}, random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Remove features with correlation greater than 0.95
corr_matrix = pd.DataFrame(X_resampled).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_resampled_dropped_corr = pd.DataFrame(X_resampled).drop(columns=to_drop)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_dropped_corr, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Run Random Forest to check feature importances
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Select the top 7 features
feature_importances = pd.DataFrame(model.feature_importances_, index=X_resampled_dropped_corr.columns, columns=['importance'])
top_features_idx = feature_importances.nlargest(8, 'importance').index
top_features_indices = [X_resampled_dropped_corr.columns.get_loc(feature) for feature in top_features_idx]
X_train_top = X_train[:, top_features_indices]
X_test_top = X_test[:, top_features_indices]

top_features = list(X.columns[top_features_indices])

# Extract feature importances
feature_importances = model.feature_importances_
sorted_idx = feature_importances.argsort()
top_features_idx = sorted_idx[-20:]

# Visualize the feature importances
plt.figure(figsize=(12, 8))
plt.barh(X.columns[top_features_idx], feature_importances[top_features_idx])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances using Random Forest')
plt.tight_layout()

# Save the plot to a file
plot_filename = "full_feature_importances.png"
plt.savefig(plot_filename)
plt.show()


# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
from sklearn.model_selection import GridSearchCV

evaluate_model(model, X_train_top, y_train, X_test_top, y_test)

# Define hyperparameters for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# # # Use GridSearchCV to find the best hyperparameters
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, verbose=0, n_jobs=-1)
rf_grid.fit(X_train_top, y_train)
rf_best_params = rf_grid.best_params_
# rf_best_params
# rf_grid = RandomForestClassifier(random_state=42, n_estimators=150,max_depth=10,min_samples_leaf=4,min_samples_split=4)
# rf_grid.fit(X_train, y_train)

# Best hyperparameters for Random Forest
# rf_best_params = rf_grid.get_params()


# Define hyperparameters for Logistic Regression
logreg_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Use GridSearchCV to find the best hyperparameters
logreg_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=5000), logreg_params, cv=5, verbose=1, n_jobs=-1)
logreg_grid.fit(X_train_top, y_train)

# logreg_grid = LogisticRegression(random_state=42, max_iter=5000)
# logreg_grid.fit(X_train, y_train)

# Best hyperparameters for Logistic Regression
logreg_best_params = logreg_grid.best_params_


# Define hyperparameters for SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to find the best hyperparameters
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, verbose=1, n_jobs=-1)
svm_grid.fit(X_train_top, y_train)
svm_best_params = svm_grid.best_params_
svm_best_params

# svm_grid = SVC(random_state=42)
# svm_grid.fit(X_train, y_train)
# Best hyperparameters for SVM
# svm_best_params = svm_grid.get_params()

# Evaluating models using the best hyperparameters

# Define hyperparameters for XGBoost
xgb_params = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Use GridSearchCV to find the best hyperparameters
xgb_grid = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                        xgb_params, cv=5, verbose=1, n_jobs=-1)
xgb_grid.fit(X_train_top, y_train)

# xgb_grid = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# xgb_grid.fit(X_train, y_train)

# Best hyperparameters for XGBoost
xgb_best_params = xgb_grid.best_params_
xgb_best_params

# Random Forest
rf_tuned = RandomForestClassifier(**rf_best_params, random_state=42)
rf_metrics_tuned = evaluate_model(rf_tuned, X_train_top, y_train, X_test_top, y_test)
y_pred_selected = rf_grid.predict(X_test_top)
conf_matrix = confusion_matrix(y_test, y_pred_selected)
conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
conf_matrix_df

# Logistic Regression
logreg_tuned = LogisticRegression(**logreg_best_params, random_state=42, max_iter=5000)
logreg_metrics_tuned = evaluate_model(logreg_tuned, X_train, y_train, X_test, y_test)

# SVM
svm_tuned = SVC(**svm_best_params, random_state=42)
svm_metrics_tuned = evaluate_model(svm_tuned, X_train, y_train, X_test, y_test)

# XGB
xgb_tuned = XGBClassifier(**xgb_best_params, random_state=42)
xgb_metrics_tuned = evaluate_model(xgb_tuned, X_train, y_train, X_test, y_test)


# Compiling the evaluation metrics into a summary dataframe
tuned_summary_df = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "SVM"],
    "Accuracy": [rf_metrics_tuned["accuracy"], logreg_metrics_tuned["accuracy"], svm_metrics_tuned["accuracy"]],
    "Precision": [rf_metrics_tuned["precision"], logreg_metrics_tuned["precision"], svm_metrics_tuned["precision"]],
    "Recall": [rf_metrics_tuned["recall"], logreg_metrics_tuned["recall"], svm_metrics_tuned["recall"]],
    "F1 Score": [rf_metrics_tuned["f1"], logreg_metrics_tuned["f1"], svm_metrics_tuned["f1"]],
    "ROC AUC": [rf_metrics_tuned["roc_auc"], logreg_metrics_tuned["roc_auc"], svm_metrics_tuned["roc_auc"]]
})


# Adding XGBoost results to the summary dataframe
tuned_summary_df = tuned_summary_df._append({
    "Model": "XGBoost",
    "Accuracy": xgb_metrics_tuned["accuracy"],
    "Precision": xgb_metrics_tuned["precision"],
    "Recall": xgb_metrics_tuned["recall"],
    "F1 Score": xgb_metrics_tuned["f1"],
    "ROC AUC": xgb_metrics_tuned["roc_auc"]
}, ignore_index=True)


tuned_summary_df['Best Parameters'] = [
    str(rf_best_params),
    str(logreg_best_params),
    str(svm_best_params),
    str(xgb_best_params)
]

# Save the tuned summary dataframe to a downloadable file
tuned_summary_df.to_excel(tuned_summary_filename, index=False)


############
# PREDICT
############

# og_df = pd.read_csv('use_data/case_study.csv')
og_df = pd.read_csv("use_data/full_ncaa_8_12.csv")
og_df = og_df[(og_df['last_season'] == 2021) & (og_df['most_recent_class'] == 'SR')]

wnba_df = pd.read_csv('use_data/all_wnba.csv')
og_df = og_df[~og_df['player_name'].isin(wnba_df['player_name'])]

case_study_df = og_df.copy()
case_study_df.columns
# selected_features = ['tot_2p', 'tot_fga', 'tot_drb', 'tot_pf', 'tot_3p%', 'tot_fg','tot_trb', 'tot_ft%', 'tot_stl', 'tot_fg%', 'tot_2p%']
# selected_features = ['adv_ws/40', 'pg_fg%', 'adv_ts%', 'pg_2p%', 'adv_ws', 'adv_stl%', 'adv_dws', 'adv_per', 'tot_stl', 'tot_drb']
top_features = ['pg_2p%', 'adv_stl%', 'pg_fg%', 'pg_pts', 'pg_sos', 'adv_trb%', 'adv_ast%', 'pg_tov']

case_study_df= case_study_df[top_features] 
# Checking for missing values in the dataset
missing_values = case_study_df.isnull().sum()

# Impute missing values with median
for column in missing_values.index:
    if missing_values[column] > 0:
        case_study_df[column].fillna(case_study_df[column].median(), inplace=True)

scaler = StandardScaler()
case_study_df = scaler.fit_transform(case_study_df)

predicted_values = rf_tuned.predict(case_study_df)
prob_values = rf_tuned.predict_proba(case_study_df)

pred_df = og_df[["player_name"]].copy()
pred_df["Predicted_Value"] = predicted_values
pred_df["Probability_Pos"]  = prob_values[:,1]
pred_df["Probability_Neg"]  = prob_values[:,0]
pred_df.sort_values(by=['Probability_Pos'],ascending=False)
# pred_df.to_excel('')

############
# PREDICT
############

# Correcting the error and reprocessing the data

model_df = pd.read_excel("model_df.xlsx")
# Reloading the data and reprocessing 

# Binning based on value
# Binning
bin_edges = [
    model_df["ws_48_pro"].min(),
    model_df["ws_48_pro"].quantile(0.33),
    model_df["ws_48_pro"].quantile(0.67),
    model_df["ws_48_pro"].max()
]
bin_labels = ["low", "medium", "high"]
model_df["ws_48_pro_bin"] = pd.cut(model_df["ws_48_pro"], bins=bin_edges, labels=bin_labels, include_lowest=True)
model_df["ws_48_pro_bin"]
model_df = model_df.drop(columns=["ws_48_pro"])

# Selecting top 10 features
X_reduced = model_df.drop(columns=['ws_48_pro_bin'])
y_reduced = model_df['ws_48_pro_bin']
X_imputed_fs = X_reduced.fillna(X_reduced.median())
rf_classifier_fs = RandomForestClassifier(random_state=42)
rf_classifier_fs.fit(X_imputed_fs, y_reduced)
feature_importances_fs = rf_classifier_fs.feature_importances_
features_df_fs = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': feature_importances_fs
}).sort_values(by='Importance', ascending=False)
top_10_features = features_df_fs.head(10)['Feature'].tolist()
X_top_10 = model_df[top_features]

# Imputation and standardization
X_top_10_imputed = X_top_10.fillna(X_top_10.median())
X_top_10_standardized = (X_top_10_imputed - X_top_10_imputed.mean()) / X_top_10_imputed.std()

# Splitting
X_train_top_10, X_test_top_10, y_train, y_test = train_test_split(X_top_10_standardized, y_reduced, test_size=0.2, random_state=42)

# Training models and storing predictions

# Logistic Regression
logistic = LogisticRegression(max_iter=5000, random_state=42).fit(X_train_top_10, y_train)
logistic_predictions = logistic.predict(X_test_top_10)

# SVC
svc = SVC(probability=True, random_state=42).fit(X_train_top_10, y_train)
svc_predictions = svc.predict(X_test_top_10)

# Random Forest
rf = RandomForestClassifier(random_state=42).fit(X_train_top_10, y_train)
rf_predictions = rf.predict(X_test_top_10)

# XGBoost
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train_top_10, y_train_encoded)
xgb_predictions = xgb_model.predict(X_test_top_10)

# Metrics
metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "SVC", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, logistic_predictions),
        accuracy_score(y_test, svc_predictions),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test_encoded, xgb_predictions)
    ],
    "Precision": [
        precision_score(y_test, logistic_predictions, average='macro'),
        precision_score(y_test, svc_predictions, average='macro'),
        precision_score(y_test, rf_predictions, average='macro'),
        precision_score(y_test_encoded, xgb_predictions, average='macro')
    ],
    "Recall": [
        recall_score(y_test, logistic_predictions, average='macro'),
        recall_score(y_test, svc_predictions, average='macro'),
        recall_score(y_test, rf_predictions, average='macro'),
        recall_score(y_test_encoded, xgb_predictions, average='macro')
    ],
    "F1 Score": [
        f1_score(y_test, logistic_predictions, average='macro'),
        f1_score(y_test, svc_predictions, average='macro'),
        f1_score(y_test, rf_predictions, average='macro'),
        f1_score(y_test_encoded, xgb_predictions, average='macro')
    ]

})



metrics_df.to_excel('3_bin.xlsx')

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Confusion matrix for the Logistic Regression model
cm = confusion_matrix(y_test, logistic_predictions, labels=['low', 'medium', 'high'])

cm.to_excel('3_bin.xlsx')
