import json
import os

notebook_path = 'notebooks/M2_trade_analysis_FIXED.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Header to include Version Banner
banner = [
    "# NBA Trade Predictor Model v2.1\n",
    "### STATUS: RECALIBRATED & ROBUST MATCHING ACTIVE\n",
    "**If you see this, the notebook is updated. Please Restart Kernel & Run All.**\n"
]
nb['cells'][0]['source'] = banner

# 2. Rename analyze_trade to analyze_trade_v2 to force a clear error if not re-run
# Also add print statement to verify normalization
for cell in nb['cells']:
    if cell.get('id') == 'trade-handler':
        source = cell['source']
        # Find def analyze_trade and change to analyze_trade_v2
        for i, line in enumerate(source):
            if "def analyze_trade(" in line:
                source[i] = line.replace("analyze_trade(", "analyze_trade_v2(")
            if 'print(f"Analyzing trade:' in line:
                source[i] = line.replace('print(f"Analyzing trade:', 'print(f"DEBUG: V2.1 ACTIVE - Analyzing trade:')
        cell['source'] = source
        break

# 3. Update User Interaction Template
for cell in nb['cells']:
    if cell.get('id') == 'user-interaction':
        source = cell['source']
        for i, line in enumerate(source):
            if "analyze_trade(" in line:
                source[i] = line.replace("analyze_trade(", "analyze_trade_v2(")
        cell['source'] = source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully forced versioning update to v2.1")
