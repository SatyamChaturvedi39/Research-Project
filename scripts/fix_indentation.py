import json

notebook_paths = [
    r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb',
    r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis_FIXED.ipynb'
]

def fix_indentation(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_lines = []
            for line in source_lines:
                # Target the specific lines with 8 spaces that should have 4
                if line.startswith('        injury_probs = np.full'):
                    new_lines.append(line.replace('        ', '    ', 1))
                    changed = True
                elif line.startswith('        results = []'):
                    new_lines.append(line.replace('        ', '    ', 1))
                    changed = True
                else:
                    new_lines.append(line)
            cell['source'] = new_lines

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Indentation fixed in {path}")
    else:
        print(f"No indentation issues found in {path}")

for p in notebook_paths:
    fix_indentation(p)
