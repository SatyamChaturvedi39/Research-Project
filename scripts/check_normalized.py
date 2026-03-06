import pandas as pd
import numpy as np
import joblib
import unicodedata

def normalize_name(name):
    if not isinstance(name, str): return ""
    # Remove accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Lowercase and strip
    return name.lower().strip()

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

# Check top scorers after normalization search
print("\n--- Searching for Stars (Normalized) ---")
proj_df['normalized_name'] = proj_df['player_name'].apply(normalize_name)

stars_to_check = ['lebron james', 'stephen curry', 'luka doncic', 'joel embiid', 'giannis antetokounmpo']
for star in stars_to_check:
    match = proj_df[proj_df['normalized_name'] == star]
    if not match.empty:
        p = match.iloc[0]['target_next_ppg']
        orig = match.iloc[0]['player_name']
        print(f"MATCH FOUND: '{orig}' -> {p:.1f} PPG")
    else:
        print(f"NO MATCH: '{star}'")

print("\n--- Top 20 Normalized Scorers ---")
top_20 = proj_df.nlargest(20, 'target_next_ppg')
for i, row in top_20.iterrows():
    print(f"{row['player_name']} ({row['normalized_name']}): {row['target_next_ppg']:.1f} PPG")
