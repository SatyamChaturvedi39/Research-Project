"""
Fetch current 2025-26 NBA rosters from ESPN API and generate team overrides.
ESPN API is public and doesn't require auth headers like NBA.com.
"""
import urllib.request
import json
import pandas as pd
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

# ESPN team data: abbreviation -> (ESPN id, ESPN slug, our dataset code)
# ESPN uses slightly different abbreviations (GS=GSW, SA=SAS, etc.)
ESPN_TEAMS = {
    "ATL": (1, "atl", "ATL"),   "BOS": (2, "bos", "BOS"),
    "BKN": (17, "bkn", "BRK"),  "CHA": (30, "cha", "CHO"),
    "CHI": (4, "chi", "CHI"),   "CLE": (5, "cle", "CLE"),
    "DAL": (6, "dal", "DAL"),   "DEN": (7, "den", "DEN"),
    "DET": (8, "det", "DET"),   "GS":  (9, "gs",  "GSW"),
    "HOU": (10, "hou", "HOU"),  "IND": (11, "ind", "IND"),
    "LAC": (12, "lac", "LAC"),  "LAL": (13, "lal", "LAL"),
    "MEM": (29, "mem", "MEM"),  "MIA": (14, "mia", "MIA"),
    "MIL": (15, "mil", "MIL"),  "MIN": (16, "min", "MIN"),
    "NO":  (3, "no",   "NOP"),  "NY":  (18, "ny",  "NYK"),
    "OKC": (25, "okc", "OKC"),  "ORL": (19, "orl", "ORL"),
    "PHI": (20, "phi", "PHI"),  "PHX": (21, "phx", "PHO"),
    "POR": (22, "por", "POR"),  "SAC": (23, "sac", "SAC"),
    "SA":  (24, "sa",  "SAS"),  "TOR": (28, "tor", "TOR"),
    "UTAH":(26, "utah","UTA"),  "WSH": (27, "wsh", "WAS"),
}

def main():
    print("Fetching rosters from ESPN API ...")
    espn_rosters = {}  # {player_full_name: our_team_code}
    failed_teams = []

    for espn_abbr, (team_id, slug, our_code) in ESPN_TEAMS.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            
            athletes = data.get('athletes', [])
            count = 0
            for athlete in athletes:
                name = athlete.get('fullName', '')
                if name:
                    espn_rosters[name] = our_code
                    count += 1
            
            sys.stdout.write(f"  {espn_abbr}({our_code}): {count}  ")
            idx = list(ESPN_TEAMS.keys()).index(espn_abbr)
            if (idx + 1) % 6 == 0:
                print()
            time.sleep(0.3)  # be polite
            
        except Exception as e:
            sys.stdout.write(f"  {espn_abbr}: FAIL  ")
            failed_teams.append(espn_abbr)

    print(f"\n\nTotal ESPN players fetched: {len(espn_rosters)}")
    if failed_teams:
        print(f"Failed teams: {failed_teams}")

    # Load our 2024-25 data
    print("\nComparing against 2024-25 dataset ...")
    try:
        df = pd.read_csv('data/processed/player_features_v2_temporal.csv')
    except FileNotFoundError:
        print("Error: Could not find data/processed/player_features_v2_temporal.csv")
        return

    latest_season = df['season'].max()
    season_df = df[df['season'] == latest_season].copy()
    season_df = season_df.drop_duplicates(subset='player_name', keep='last')

    our_players = dict(zip(season_df['player_name'], season_df['team']))

    # Match and find overrides
    overrides = {}
    matched = 0
    unmatched = []

    for our_name, our_team in our_players.items():
        # Try exact match
        if our_name in espn_rosters:
            espn_team = espn_rosters[our_name]
            matched += 1
            if espn_team != our_team:
                overrides[our_name] = espn_team
        else:
            # Try case-insensitive
            found = False
            name_lower = our_name.lower()
            for espn_name, espn_team in espn_rosters.items():
                if espn_name.lower() == name_lower:
                    matched += 1
                    found = True
                    if espn_team != our_team:
                        overrides[our_name] = espn_team
                    break
            
            if not found:
                # Try partial match (first + last name)
                parts = our_name.split()
                if len(parts) >= 2:
                    first_last = (parts[0].lower(), parts[-1].lower())
                    for espn_name, espn_team in espn_rosters.items():
                        e_parts = espn_name.split()
                        if len(e_parts) >= 2:
                            e_first_last = (e_parts[0].lower(), e_parts[-1].lower())
                            if first_last == e_first_last:
                                matched += 1
                                found = True
                                if espn_team != our_team:
                                    overrides[our_name] = espn_team
                                break
                
                if not found:
                    unmatched.append(our_name)

    print(f"Matched: {matched}/{len(our_players)} players")
    print(f"Overrides (team changes): {len(overrides)}")
    print(f"Unmatched: {len(unmatched)} (likely retired/overseas/G-League/name mismatch)")

    # Write the overrides
    output = {
        "_meta": {
            "season": "2025-26",
            "last_updated": pd.Timestamp.today().strftime('%Y-%m-%d'),
            "description": "Team overrides auto-generated from ESPN roster API",
            "source": "site.api.espn.com",
            "matched_players": matched,
            "overrides_count": len(overrides),
            "note": "Only players from the 2024-25 dataset who changed teams are included."
        },
        "overrides": dict(sorted(overrides.items()))
    }

    output_path = 'data/team_overrides_2025_26.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {len(overrides)} overrides to {output_path}")

if __name__ == "__main__":
    main()
