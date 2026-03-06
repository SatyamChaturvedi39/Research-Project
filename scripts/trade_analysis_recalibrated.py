import pandas as pd
import numpy as np
import joblib
import os

# 1. Load Data and Models
print("Loading data and models...")
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

# recalibrated std devs for MC
std_devs = {
    'target_next_ppg': 3.0,
    'target_next_rpg': 1.2,
    'target_next_apg': 1.0,
    'target_next_mpg': 3.5,
    'target_next_ts_pct': 0.03
}

# 4. Corrected Fit Heuristics and Win Prediction
def calculate_fit_penalty(roster):
    penalty = 0
    pos_counts = roster['position'].value_counts()
    if pos_counts.get('PG', 0) > 3: penalty += 0.05
    if pos_counts.get('C', 0) > 3: penalty += 0.03

    # Recalibrated threshold: 75 PPG for top 3 scorers
    top_scorers = roster.nlargest(3, 'target_next_ppg')['target_next_ppg'].sum()
    if top_scorers > 75:
        penalty += 0.08

    return penalty

def calculate_team_score(roster, sampled_stats):
    top_8_stats = sampled_stats.nlargest(8, 'target_next_ppg')
    top_8_ppg = top_8_stats['target_next_ppg'].sum()
    efficiency_bonus = top_8_stats['target_next_ts_pct'].mean() * 10

    fit_penalty = calculate_fit_penalty(roster)

    score = (top_8_ppg + efficiency_bonus) * (1 - fit_penalty)
    return score

def predict_wins(score):
    # Recalibrated formula: Mean score ~113.7 -> 41 wins (shifted from 118.7 to account for MC variance)
    wins = (score - 113.7) * 1.2 + 41
    return np.clip(wins, 5, 75)

def wins_to_playoff_prob(wins):
    # Reduced sensitivity
    if wins >= 42:
        return np.clip((wins - 42) * 0.05 + 0.5, 0.5, 1.0)
    else:
        return np.clip(0.5 - (42 - wins) * 0.05, 0.0, 0.5)

# 5. Monte Carlo Engine
def run_simulation(roster, n_sims=1000):
    injury_probs = np.full(len(roster), 0.05) 

    results = []
    for _ in range(n_sims):
        sampled_stats = roster.copy()
        for col, std in std_devs.items():
            sampled_stats[col] = np.maximum(0, np.random.normal(roster[col], std))

        is_injured = np.random.random(len(roster)) < injury_probs
        sampled_stats.loc[is_injured, 'target_next_ppg'] *= 0.6

        score = calculate_team_score(roster, sampled_stats)
        wins = predict_wins(score)
        results.append(wins)

    return np.array(results)

# Verification Logic
print("\n--- Verifying Recalibrated distributed wins ---")
all_team_wins = []
for team in proj_df['team'].unique():
    roster = proj_df[proj_df['team'] == team]
    if len(roster) >= 8:
        expected_wins = run_simulation(roster, n_sims=100).mean()
        all_team_wins.append((team, expected_wins))

all_team_wins.sort(key=lambda x: x[1], reverse=True)
print(f"Mean Wins League Wide: {np.mean([w[1] for w in all_team_wins]):.2f}")
print("\nTop 5 Teams:")
for t, w in all_team_wins[:5]:
    print(f"{t}: {w:.1f} wins")

print("\nBottom 5 Teams:")
for t, w in all_team_wins[-5:]:
    print(f"{t}: {w:.1f} wins")
