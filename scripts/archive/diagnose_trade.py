import pandas as pd
import numpy as np
import joblib
import json

# Setup paths (relative to root)
player_csv = 'data/processed/player_features_v2_temporal.csv'
model_path = 'models/player_multioutput_v2.pkl'
features_path = 'models/feature_names_v2.txt'
targets_path = 'models/target_names_v2.txt'

# Load Data and Models
player_df = pd.read_csv(player_csv)
player_model = joblib.load(model_path)

with open(features_path, 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]
with open(targets_path, 'r') as f:
    target_cols = [line.strip() for line in f.readlines()]

# Get latest player data
latest_player_data = player_df.sort_values(['player_name', 'season']).groupby('player_name').last().reset_index()

# Inference
X_inference = latest_player_data[feature_cols].fillna(0)
projections = player_model.predict(X_inference)

proj_df = pd.DataFrame(projections, columns=target_cols)
proj_df['player_name'] = latest_player_data['player_name']
proj_df['team'] = latest_player_data['team']
proj_df['position'] = latest_player_data['position']

# Diagnostic: LeBron vs Curry
lebron = proj_df[proj_df['player_name'].str.contains('LeBron James', na=False)]
curry = proj_df[proj_df['player_name'].str.contains('Stephen Curry', na=False)]

print("\n--- LeBron James Stats ---")
print(lebron[target_cols + ['team', 'position']])

print("\n--- Stephen Curry Stats ---")
print(curry[target_cols + ['team', 'position']])

# Check teams
print("\nLakers Roster Sample:")
print(proj_df[proj_df['team'] == 'LAL']['player_name'].head())

print("\nWarriors Roster Sample:")
print(proj_df[proj_df['team'] == 'GSW']['player_name'].head())
