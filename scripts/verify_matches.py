import pandas as pd
import re

player_df = pd.read_csv(r'c:\MSAIML\research_proj\Research-Project\data\processed\player_features_v2_temporal.csv')
injury_df = pd.read_csv(r'c:\MSAIML\research_proj\Research-Project\data\processed\player_injury_score_2020_2025_cleaned.csv')

def clean_name(x):
    return re.sub(r'\(.*?\)', '', str(x)).strip()

player_names = set(player_df['player_name'].str.strip())
injury_names_raw = set(injury_df['Name'])
injury_names_cleaned = set(injury_df['Name'].apply(clean_name))

overlap_raw = player_names.intersection(injury_names_raw)
overlap_cleaned = player_names.intersection(injury_names_cleaned)

print(f"Unique player names in features: {len(player_names)}")
print(f"Unique player names in injury (raw): {len(injury_names_raw)}")
print(f"Overlap (raw): {len(overlap_raw)}")
print(f"Overlap (cleaned): {len(overlap_cleaned)}")

if len(overlap_cleaned) > len(overlap_raw):
    print("Name cleaning improved matching!")
else:
    print("Name cleaning did not improve matching or it was already optimal.")

# Sample of names that still don't match
unmatched_injury = injury_names_cleaned - player_names
print(f"Sample unmatched injury names: {list(unmatched_injury)[:10]}")
