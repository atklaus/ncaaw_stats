import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#########################
# SET RUN TYPE
#########################
n=11
tuned_summary_filename = "3_model_evaluation_11_feat.xlsx"
# Load the data from the provided Excel file
df = pd.read_excel("model_df.xlsx")
cols_to_keep = [col for col in df.columns if col.startswith('tot_')] + ['ws_48_pro']
df = df[cols_to_keep]


#########################
# MODEL
#########################

stats = df.describe()

# Checking for missing values in the dataset
missing_values = df.isnull().sum()

stats, missing_values

# Impute missing values with median
for column in missing_values.index:
    if missing_values[column] > 0:
        df[column].fillna(df[column].median(), inplace=True)

# Confirm that there are no missing values left
missing_after_imputation = df.isnull().sum()

import numpy as np

# Compute the correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
df_dropped_corr = df.drop(columns=to_drop)

to_drop, df_dropped_corr.shape

from sklearn.preprocessing import StandardScaler

# Separating the features and target variable
X = df_dropped_corr.drop(columns=["ws_48_pro"])
y = (df_dropped_corr["ws_48_pro"] > 0).astype(int)  # Binary classification (1 if ws_48_pro > 0 else 0)
# y = pd.qcut(df['ws_48_pro'], q=3, labels=['Low', 'Medium', 'High'])

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Run a random forest to check feature importances
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
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# Print the best parameters
print(grid_search.best_params_)

# Use the best model
best_model = grid_search.best_estimator_

# Extract feature importances
feature_importances = best_model.feature_importances_
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

# n=11
top_features = X.columns[sorted_idx][n:]
feature_importances = pd.DataFrame(best_model.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
X_top = X[feature_importances.nlargest(n, 'importance').index]

# Split the data into training and test sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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

# Define hyperparameters for Random Forest
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Use GridSearchCV to find the best hyperparameters
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, verbose=1, n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Best hyperparameters for Random Forest
rf_best_params = rf_grid.best_params_
rf_best_params

# Define hyperparameters for Logistic Regression
logreg_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Use GridSearchCV to find the best hyperparameters
logreg_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=5000), logreg_params, cv=5, verbose=1, n_jobs=-1)
logreg_grid.fit(X_train, y_train)

# Best hyperparameters for Logistic Regression
logreg_best_params = logreg_grid.best_params_
logreg_best_params


# Define hyperparameters for SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to find the best hyperparameters
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, verbose=1, n_jobs=-1)
svm_grid.fit(X_train, y_train)

# Best hyperparameters for SVM
svm_best_params = svm_grid.best_params_
svm_best_params
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
xgb_grid.fit(X_train, y_train)

# Best hyperparameters for XGBoost
xgb_best_params = xgb_grid.best_params_
xgb_best_params


# Random Forest
rf_tuned = RandomForestClassifier(**rf_best_params, random_state=42)
rf_metrics_tuned = evaluate_model(rf_tuned, X_train, y_train, X_test, y_test)

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

