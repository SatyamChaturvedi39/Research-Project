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

print("\n--- Statistics for Top Predicted Scorers ---")
top_scorers = proj_df.nlargest(10, 'target_next_ppg')
print(top_scorers[['player_name', 'target_next_ppg', 'target_next_apg']])

print("\n--- Specific Checks ---")
for name in ['Stephen Curry', 'LeBron James', 'Kevin Durant', 'Joel Embiid', 'Luka Doncic']:
    stats = proj_df[proj_df['player_name'] == name]
    if not stats.empty:
        p = stats.iloc[0]['target_next_ppg']
        a = stats.iloc[0]['target_next_apg']
        print(f"{name}: {p:.1f} PPG, {a:.1f} APG")
    else:
        print(f"{name} NOT FOUND with exact name.")
