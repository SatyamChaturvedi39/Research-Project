import json
import re

notebook_paths = [
    r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis.ipynb',
    r'c:\MSAIML\research_proj\Research-Project\notebooks\M2_trade_analysis_FIXED.ipynb'
]

def clean_indentation(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            # 1. Join to single string and replace tabs with 4 spaces
            source = "".join(cell.get('source', []))
            source = source.replace('\t', '    ')
            
            # 2. Split into lines
            lines = source.split('\n')
            new_lines = []
            
            in_run_sim = False
            for line in lines:
                # 3. Handle specific run_simulation normalization
                if 'def run_simulation(roster, n_sims=1000):' in line:
                    in_run_sim = True
                    new_lines.append(line.rstrip())
                    continue
                
                if in_run_sim:
                    # If we hit another function or top level comment, we might be out
                    # but for this specific notebook we know the structure
                    if line.startswith('def ') or (line.startswith('#') and not line.startswith('# ')):
                        # Simple heuristic for end of function in this notebook
                        # but actually we can just clean everything
                        pass
                    
                    # Strip trailing space and re-indent if it seems to be body
                    stripped = line.lstrip()
                    if stripped:
                        # If it's a body line, force 4 spaces unless it's deeper (loop)
                        # Let's just normalize leading spaces to 4 if it was 4 or more
                        leading_space_count = len(line) - len(stripped)
                        if leading_space_count >= 4:
                            # Keep relative indentation for loops
                            normalized_indent = " " * (leading_space_count)
                            new_lines.append(normalized_indent + stripped.rstrip())
                        else:
                            new_lines.append(line.rstrip())
                    else:
                        # Empty line in function
                        new_lines.append("")
                else:
                    new_lines.append(line.rstrip())

            # 4. Final join and ensure newlines
            source_final = "\n".join(new_lines)
            
            # Explicit fix for the known problematic results = [] line
            # It should have exactly 4 spaces if it follows injury_probs
            source_final = source_final.replace('injury_probs = np.full(len(roster), 0.05) # Placeholder injury risk\n    \n    results = []', 
                                              'injury_probs = np.full(len(roster), 0.05) # Placeholder injury risk\n    \n    results = []')
            # Actually let's just do a regex for the whole block
            pattern = r'def run_simulation\(roster, n_sims=1000\):\s+injury_probs = np\.full\(len\(roster\), 0\.05\) # Placeholder injury risk\s+results = \[\]'
            # (Matches without the intermediate empty line in case it's gone)
            
            # Re-split for the source list format
            final_list = [l + "\n" for l in new_lines]
            if final_list and final_list[-1] == "\n":
                final_list.pop()
            else:
                if final_list:
                    final_list[-1] = final_list[-1].rstrip("\n")
            
            cell['source'] = final_list

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Aggressive cleaning applied to {path}")

for p in notebook_paths:
    clean_indentation(p)
