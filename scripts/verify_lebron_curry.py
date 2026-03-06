import pandas as pd
import numpy as np
import joblib
import unicodedata

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
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
proj_df['position'] = latest_player_data['position']
proj_df['normalized_name'] = proj_df['player_name'].apply(normalize_name)

# Helper functions from notebook
def calculate_fit_penalty(roster):
    penalty = 0
    pos_counts = roster['position'].value_counts()
    if pos_counts.get('PG', 0) > 3: penalty += 0.05
    if pos_counts.get('C', 0) > 3: penalty += 0.03
    top_scorers = roster.nlargest(3, 'target_next_ppg')['target_next_ppg'].sum()
    if top_scorers > 75: penalty += 0.08
    return penalty

def calculate_team_score(roster, sampled_stats):
    top_8_stats = sampled_stats.nlargest(8, 'target_next_ppg')
    top_8_ppg = top_8_stats['target_next_ppg'].sum()
    efficiency_bonus = top_8_stats['target_next_ts_pct'].mean() * 10
    fit_penalty = calculate_fit_penalty(roster)
    score = (top_8_ppg + efficiency_bonus) * (1 - fit_penalty)
    return score

def predict_wins(score):
    wins = (score - 113.7) * 1.2 + 41
    return np.clip(wins, 5, 75)

# Verification Logic: LeBron for Curry
team_a = 'LAL'
team_b = 'GSW'
out_a = ['lebron james']
out_b = ['stephen curry']

roster_a = proj_df[proj_df['team'] == team_a].copy()
roster_b = proj_df[proj_df['team'] == team_b].copy()

# Execute trade (normalized)
p_a = roster_a[roster_a['normalized_name'].isin(out_a)]
p_b = roster_b[roster_b['normalized_name'].isin(out_b)]

print(f"LAL outgoing: {len(p_a)} players, GSW outgoing: {len(p_b)} players")

if len(p_a) > 0 and len(p_b) > 0:
    post_a = pd.concat([roster_a[~roster_a['normalized_name'].isin(out_a)], p_b])
    
    pre_score = calculate_team_score(roster_a, roster_a)
    post_score = calculate_team_score(post_a, post_a)
    
    pre_wins = predict_wins(pre_score)
    post_wins = predict_wins(post_score)
    
    print(f"\nLakers Impact (Deterministic):")
    print(f"Pre-Trade: {pre_wins:.2f} wins")
    print(f"Post-Trade: {post_wins:.2f} wins")
    print(f"Change: {post_wins - pre_wins:+.2f} wins")
else:
    print("ERROR: Players NOT found during trade simulation.")
