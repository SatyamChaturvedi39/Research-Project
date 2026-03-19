import pandas as pd
import numpy as np
import joblib

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

def calculate_team_score(roster):
    top_8_stats = roster.nlargest(8, 'target_next_ppg')
    score = top_8_stats['target_next_ppg'].sum() + top_8_stats['target_next_ts_pct'].mean() * 10
    return score

team_scores = []
for team in proj_df['team'].unique():
    roster = proj_df[proj_df['team'] == team]
    if len(roster) >= 8:
        team_scores.append(calculate_team_score(roster))

print("Mean Score:", np.mean(team_scores))
print("Min Score:", np.min(team_scores))
print("Max Score:", np.max(team_scores))
print("Std Score:", np.std(team_scores))
