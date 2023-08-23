import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from datetime import datetime, timedelta
import os
import csv
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt


os.chdir('/Users/bosnianthundaa/Documents/Basketball Prediction Project/Game Table URLs')

# Directory where your CSV files are located
csv_directory = '/Users/bosnianthundaa/Documents/Basketball Prediction Project/Game Table URLs'

# Prefix of the CSV files
file_prefix = 'all_game_logs - '

# Get a list of all CSV files with the specified prefix in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.startswith(file_prefix) and file.endswith('.csv')]

# Initialize an empty DataFrame to hold the combined data
combined_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined DataFrame
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    data = pd.read_csv(file_path)
    combined_data = combined_data.append(data, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_data.to_csv('all_game_logs_combined.csv', index=False)

print("CSV files appended and saved as 'combined_data.csv'")


#### Change PHX to PHO ####
combined_data.replace('PHX', 'PHO', inplace=True)


#### Re-do home game and win columns
combined_data.drop(['home_game', 'win'], axis=1, inplace=True)

combined_data['home_game'] = combined_data.apply(lambda row: 1 if row['home_team'] == row['team_of_stats'] else 0, axis=1)
combined_data['win'] = combined_data.apply(lambda row: 1 if row['winning_team'] == row['team_of_stats'] else 0, axis=1)


### Adding in altitude

# Altitude dictionary
nba_cities_altitudes = {
    "ATL": 1050,    # Altitude in feet
    "BOS": 141,      # Altitude in feet
    "BRK": 30,     # Altitude in feet
    "CHO": 751,   # Altitude in feet
    "CHI": 594,     # Altitude in feet
    "CLE": 653,   # Altitude in feet
    "DAL": 430,      # Altitude in feet
    "DEN": 5280,     # Altitude in feet
    "DET": 600,     # Altitude in feet
    "GSW": 13, # Altitude in feet (San Francisco)
    "HOU": 43,      # Altitude in feet
    "IND" : 715,
    "LAC": 233,          # Altitude in feet (Los Angeles)
    "LAL": 233, # Altitude in feet (Los Angeles)
    "MEM": 337,     # Altitude in feet
    "MIA": 6,         # Altitude in feet
    "MIL": 617,   # Altitude in feet
    "MIN": 830,   # Altitude in feet (Minneapolis)
    "NOP": 7,   # Altitude in feet
    "NYK": 33,     # Altitude in feet
    "OKC": 1200, # Altitude in feet
    "ORL": 82,      # Altitude in feet
    "PHI": 39, # Altitude in feet
    "PHO": 1086,    # Altitude in feet
    "POR": 43,     # Altitude in feet
    "SAC": 30,   # Altitude in feet
    "SAS": 650, # Altitude in feet
    "TOR": 249,     # Altitude in feet
    "UTA": 4226,       # Altitude in feet
    "WAS": 72   # Altitude in feet
}

combined_data['altitude'] = combined_data['home_team'].map(nba_cities_altitudes)

#### Create back-to-back variable (1/0) ####

# Sort the DataFrame by team and date
combined_data.sort_values(['team_of_stats', 'date'], inplace=True)

# Calculate the difference in days between consecutive dates for each team
combined_data['date'] = pd.to_datetime(combined_data['date'])
combined_data['Days_diff'] = combined_data.groupby('team_of_stats')['date'].diff().fillna(pd.Timedelta(days=0))

# Create the 'back-to-back' column based on the Days_diff values
combined_data['back-to-back'] = combined_data['Days_diff'].dt.days == 1

# Convert boolean values to 1 and 0
combined_data['back-to-back'] = combined_data['back-to-back'].astype(int)

# Drop the 'Days_diff' column if no longer needed
combined_data.drop(columns='Days_diff', inplace=True)

#### Create wins in last 10 games ####
# Sort the DataFrame by team and date
combined_data.sort_values(['team_of_stats', 'date'], inplace=True)

# Calculate the rolling sum of wins in the last 10 games
combined_data['last_10_wins'] = combined_data.groupby('team_of_stats')['win'].rolling(window=10, min_periods=1).sum().reset_index(level=0, drop=True)

combined_data.to_csv('combined_data_final.csv')
########################

#### EDA on dataset ####

combined_data = pd.read_csv('all_game_logs_combined.csv')
print(combined_data.describe())

#### Checking for duplicates ####

duplicate_rows = combined_data.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())
combined_data = combined_data.drop_duplicates(keep='first')

#### Checking for NULLs ####
# Check for NULL values in each column
null_values_per_column = combined_data.isnull().sum()

# Display NULL values count per column
print("NULL values per column:")
print(null_values_per_column)

# NO NULLs #

#### Boxplots for each column ####

for column in combined_data.columns:
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(data[column]):
        plt.figure(figsize=(8, 6))
        plt.boxplot(data[column], vert=False)
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.show()

#### Violin plots for each column ####

for column in combined_data.columns:
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(data[column]):
        plt.figure(figsize=(8, 6))
        sns.violinplot(data[column])
        plt.title(f'Violin Plot of {column}')
        plt.xlabel(column)
        plt.show()

#### Correlation between variables ####

# Calculate correlation matrix
correlation_matrix = combined_data.corr()

# Set up the figure size
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)

# Adjust fontsize of the labels and values
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


# Save the heatmap as a PNG file
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()



####### MODEL CREATION ########

##############################
### Model 1: Random Forest ###
##############################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('combined_data_final.csv')

column_names = data.columns.tolist()

vars_to_remove = ['Unnamed: 0', 'home_team', 'away_team', 'date', 'team_of_stats', 'winning_team', 'win']

for value in vars_to_remove:
    column_names.remove(value)

feature_cols = column_names
# 'feature_cols' are the columns I'm using as features
# 'target_col' is the column indicating win/loss

X = data[feature_cols]
y = data['win']

## Split your data into training and testing sets to evaluate your model's performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Create a Random Forest model using the training data and train the model.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## Predict outcomes on the test data and evaluate the model's performance using appropriate metrics.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

feature_importances = model.feature_importances_
print("Feature Importances:", feature_importances)


#### Hyperparameter Tuning ####
## Improve model by tuning hyperparameters using techniques like Grid Search or Random Search.

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    # Add more hyperparameters to tune
}

## Perform Grid Search ##
## Grid Search exhaustively tries all possible combinations of hyperparameters from the defined grid. It's suitable when you have a relatively small number of hyperparameters to tune.

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model_grid= grid_search.best_estimator_
best_params_grid = grid_search.best_params_

## Evaluate the Best Model
## After tuning, evaluate the performance of the best model using the test set.

y_pred = best_model_grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Model Accuracy with grid search:", accuracy)
print("Best Model Classification Report with grid search:\n", report)


## Perform Random Search ##
## Random Search randomly samples a defined number of hyperparameter combinations from the specified ranges. It's more efficient when you have a large hyperparameter space.

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_model_random = random_search.best_estimator_
best_params_random = random_search.best_params_


## Evaluate the Best Model:
## After tuning, evaluate the performance of the best model using the test set.

y_pred = best_model_random.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Model Accuracy with random search:", accuracy)
print("Best Model Classification Report with random search:\n", report)


####################################
### Model 2: Logistic Regression ###
####################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess your data

data = pd.read_csv('combined_data_final.csv')

column_names = data.columns.tolist()

vars_to_remove = ['Unnamed: 0', 'home_team', 'away_team', 'date', 
                  'team_of_stats', 'winning_team', 'win', 'total_assists',
                  'total_rebounds', 'total_steals' ,'total_blocks',
                  'total_turnovers']

for value in vars_to_remove:
    column_names.remove(value)

feature_cols = column_names
# 'feature_cols' are the columns I'm using as features
# 'target_col' is the column indicating win/loss

X = data[feature_cols]
y = data['win']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Get the feature names and coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Set a threshold for feature selection
threshold = .009 # Adjust as needed

# Select features with coefficients above the threshold
selected_features = [feature for feature, coef in zip(feature_names, coefficients) if abs(coef) > threshold]

print("Selected Features:", selected_features)

# Create new datasets with only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train and evaluate a model using only the selected features
selected_model = LogisticRegression()
selected_model.fit(X_train_selected, y_train)
accuracy = selected_model.score(X_test_selected, y_test)
print("Accuracy with Selected Features:", accuracy)

# Calculate accuracy

## Input new set of numbers

import numpy as np

# Assuming new_data is a dictionary or a list containing new input values
# Example: new_data = {'total_points': 110, 'total_rebounds': 45, 'home_team': 1, ...}

# Convert new_data into a suitable format (e.g., DataFrame or NumPy array)
# For example, if new_data is a dictionary:
new_stats_to_test = {
    'team_true_shooting': 0.8,
    'team_effective_shooting': 0.8,
    'team_rebound_rate': 57,
    'team_turnover_rate': 4,
    'team_off_rtg': 118,
    'home_game': 1,
    'last_10_wins': 6,
    }
new_data_array = np.array([[new_stats_to_test['team_true_shooting'], 
                            new_stats_to_test['team_effective_shooting'], 
                            new_stats_to_test['team_rebound_rate'], 
                            new_stats_to_test['team_turnover_rate'], 
                            new_stats_to_test['team_off_rtg'],
                            new_stats_to_test['home_game'], 
                            new_stats_to_test['last_10_wins']]])

# Predict the outcome (0 or 1)
predicted_outcome = selected_model.predict(new_data_array)

print("Predicted Outcome:", predicted_outcome)

# Predict the probabilities for both classes (0 and 1)
predicted_probabilities = selected_model.predict_proba(new_data_array)

# Probability of class 0 (loss)
probability_0 = predicted_probabilities[0, 0]

# Probability of class 1 (win)
probability_1 = predicted_probabilities[0, 1]

print("Probability of Losing (Class 0):", probability_0)
print("Probability of Winning (Class 1):", probability_1)

