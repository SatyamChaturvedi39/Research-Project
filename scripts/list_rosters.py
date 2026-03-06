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

lal_roster = proj_df[proj_df['team'] == 'LAL']['player_name'].tolist()
gsw_roster = proj_df[proj_df['team'] == 'GSW']['player_name'].tolist()

print("\n--- Lakers Roster ---")
for p in sorted(lal_roster):
    print(f"'{p}'")

print("\n--- Warriors Roster ---")
for p in sorted(gsw_roster):
    print(f"'{p}'")

# Check predicted PPG for samples
lebron_ppg = proj_df[proj_df['player_name'].str.contains('LeBron', na=False)]['target_next_ppg'].values
print(f"\nLeBron PPG Projections: {lebron_ppg}")

curry_ppg = proj_df[proj_df['player_name'].str.contains('Curry', na=False)]['target_next_ppg'].values
print(f"Curry PPG Projections: {curry_ppg}")
