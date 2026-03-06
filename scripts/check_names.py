import pandas as pd
import numpy as np
import joblib

# Load Data and Models
player_df = pd.read_csv('data/processed/player_features_v2_temporal.csv')
player_model = joblib.load('models/player_multioutput_v2.pkl')

with open('models/feature_names_v2.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]
with open('models/target_names_v2.txt', 'r') as f:
    target_cols = [line.strip() for line in f.readlines()]

latest_player_data = player_df.sort_values(['player_name', 'season']).groupby('player_name').last().reset_index()

X_inference = latest_player_data[feature_cols].fillna(0)
projections = player_model.predict(X_inference)

proj_df = pd.DataFrame(projections, columns=target_cols)
proj_df['player_name'] = latest_player_data['player_name']
proj_df['team'] = latest_player_data['team']
proj_df['position'] = latest_player_data['position']

# Find exact names
lebrons = proj_df[proj_df['player_name'].str.contains('LeBron', case=False, na=False)]['player_name'].tolist()
currys = proj_df[proj_df['player_name'].str.contains('Curry', case=False, na=False)]['player_name'].tolist()

print("Exact LeBron matches:", lebrons)
print("Exact Curry matches:", currys)

# Find unique team IDs
print("\nUnique Teams in Data:", sorted(proj_df['team'].unique().tolist()))

# Check LAL and GSW existence
lal_exists = 'LAL' in proj_df['team'].unique()
gsw_exists = 'GSW' in proj_df['team'].unique()
print(f"LAL in data: {lal_exists}")
print(f"GSW in data: {gsw_exists}")
