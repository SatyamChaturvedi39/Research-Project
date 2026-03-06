import json

notebook_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb'
fixed_path = r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis_FIXED.ipynb'

def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Fix the ):[:, 1] syntax error
            if '):[:, 1]' in source:
                source = source.replace('):[:, 1]\n', '')
                source = source.replace('):[:, 1]', '')
            
            # Ensure proper indentation for placeholder
            if 'injury_probs = np.full(len(roster), 0.05)' in source:
                 if '    injury_probs =' not in source:
                     source = source.replace('injury_probs = np.full(len(roster), 0.05)', '    injury_probs = np.full(len(roster), 0.05)')

            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1] == '\n':
                cell['source'].pop()
            else:
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

patch_file(notebook_path)
patch_file(fixed_path)

print("Syntax fix applied to both notebooks.")
