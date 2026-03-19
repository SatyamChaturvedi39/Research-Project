import json
import os

notebook_path = 'notebooks/M2_trade_analysis_FIXED.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 2 (Imports) to include unicodedata
for cell in nb['cells']:
    if cell.get('id') == 'imports':
        cell['source'] = [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "import os\n",
            "import json\n",
            "import unicodedata\n",
            "from tqdm import tqdm\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "sns.set_style('whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)"
        ]
        break

# 2. Add Normalization Helper to a new cell or update Generate Predictions
normalization_code = [
    "def normalize_name(name):\n",
    "    if not isinstance(name, str): return \"\"\n",
    "    # Remove accents and normalize to lowercase\n",
    "    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')\n",
    "    return name.lower().strip()\n",
    "\n",
    "proj_df['normalized_name'] = proj_df['player_name'].apply(normalize_name)\n",
    "print(\"Player names normalized for robust matching.\")"
]

# Insert normalization logic after Generate Predictions (Cell 3, index 4 depends on nb structure)
# Let's find index of 'generate-predictions' and insert after it
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'generate-predictions':
        # Add normalization line to the end of this cell
        cell['source'].extend([
            "\n",
            "# Normalize player names for case-insensitive matching\n",
            "def normalize_name(name):\n",
            "    if not isinstance(name, str): return \"\"\n",
            "    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')\n",
            "    return name.lower().strip()\n",
            "\n",
            "proj_df['normalized_name'] = proj_df['player_name'].apply(normalize_name)\n",
            "print(\"Names normalized.\")\n"
        ])
        break

# 3. Update analyze_trade to be robust
new_trade_handler = [
    "# 6. Trade Scenario Handler\n",
    "\n",
    "def analyze_trade(team_a_id, team_b_id, outgoing_a, outgoing_b):\n",
    "    \"\"\"\n",
    "    Executes trade analysis between two teams with robust name matching.\n",
    "    \"\"\"\n",
    "    # Normalize inputs\n",
    "    outgoing_a = [normalize_name(p) for p in outgoing_a]\n",
    "    outgoing_b = [normalize_name(p) for p in outgoing_b]\n",
    "    \n",
    "    # Get base rosters\n",
    "    roster_a = proj_df[proj_df['team'] == team_a_id].copy()\n",
    "    roster_b = proj_df[proj_df['team'] == team_b_id].copy()\n",
    "    \n",
    "    if roster_a.empty: print(f\"Warning: No players found for Team {team_a_id}\")\n",
    "    if roster_b.empty: print(f\"Warning: No players found for Team {team_b_id}\")\n",
    "\n",
    "    # Execute trade using normalized names\n",
    "    players_a = roster_a[roster_a['normalized_name'].isin(outgoing_a)]\n",
    "    players_b = roster_b[roster_b['normalized_name'].isin(outgoing_b)]\n",
    "    \n",
    "    # Diagnostic messaging\n",
    "    for p_name in outgoing_a:\n",
    "        if p_name not in roster_a['normalized_name'].values:\n",
    "            print(f\"Warning: Player '{p_name}' not found on {team_a_id} roster!\")\n",
    "    for p_name in outgoing_b:\n",
    "        if p_name not in roster_b['normalized_name'].values:\n",
    "            print(f\"Warning: Player '{p_name}' not found on {team_b_id} roster!\")\n",
    "\n",
    "    post_roster_a = pd.concat([roster_a[~roster_a['normalized_name'].isin(outgoing_a)], players_b])\n",
    "    post_roster_b = pd.concat([roster_b[~roster_b['normalized_name'].isin(outgoing_b)], players_a])\n",
    "\n",
    "    print(f\"Analyzing trade: {len(players_a)} players from {team_a_id} <-> {len(players_b)} players from {team_b_id}\")\n",
    "\n",
    "    results = {}\n",
    "    for label, pre, post in [('Team A', roster_a, post_roster_a), ('Team B', roster_b, post_roster_b)]:\n",
    "        print(f\"Simulating {label}...\")\n",
    "        pre_results = run_simulation(pre)\n",
    "        post_results = run_simulation(post)\n",
    "        results[label] = (pre_results, post_results)\n",
    "\n",
    "    return results"
]

for cell in nb['cells']:
    if cell.get('id') == 'trade-handler':
        cell['source'] = new_trade_handler
        break

# 4. Add Search Player Tool at the end
search_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "player-search",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Utility: Search for Player Names\n",
        "def search_players(query):\n",
        "    query = normalize_name(query)\n",
        "    matches = proj_df[proj_df['normalized_name'].str.contains(query, na=False)]\n",
        "    if matches.empty:\n",
        "        print(\"No matches found.\")\n",
        "    else:\n",
        "        print(matches[['player_name', 'team', 'position', 'target_next_ppg']])\n",
        "\n",
        "# Example usage:\n",
        "# search_players('lebron')"
    ]
}
nb['cells'].append(search_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully applied reliability fixes to M2_trade_analysis_FIXED.ipynb")
