import re, json

with open('app_part1.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()

with open('app_part2.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()

with open('app_part3.txt', 'r', encoding='utf-8') as f:
    text3 = f.read()

with open('.git/../app.py', 'r', encoding='utf-8') as f:
    repo_app = f.readlines()

def clean_lines(text):
    out = []
    lines = text.strip().split('\n')
    for line in lines:
        if '<line_number>:' in line or re.match(r'^\d+:', line):
            content = line.split(': ', 1)
            if len(content) == 2:
                out.append(content[1])
            else:
                out.append("")
    return out

p1 = clean_lines(text1) # 1 to 200
p2 = clean_lines(text2) # 190 to 500
p3 = clean_lines(text3) # 621 to 1420

# We need the user's lines 501 to 620!
# We know repo_app lines 445 to 566 correspond to that.
# Actually let's just use the repo version and modify it.

