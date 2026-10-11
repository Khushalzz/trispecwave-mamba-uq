import re
import os

files = [
    'docs/section_ii_literature_survey_clean.md',
    'docs/section_iii_methodology.md',
    'docs/section_iv_results_and_discussion.md',
    'docs/section_v_research_gaps_and_future_horizons.md'
]

forbidden_words = ['sih', 'smart india hackathon', 'ps-2', 'ps2', 'problem statement']

for file_path in files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\n--- Checking {file_path} ---")
    
    # 1. Check forbidden words (case-insensitive)
    for word in forbidden_words:
        matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))
        if matches:
            print(f"  [FORBIDDEN WORD FOUND] '{word}' matched {len(matches)} times:")
            for m in matches[:3]:
                snippet = text[max(0, m.start()-30):min(len(text), m.end()+30)].replace('\n', ' ')
                print(f"    ...{snippet}...")

    # 2. Check KaTeX potential pitfalls
    tags = list(re.finditer(r'\\tag\*?\{[^}]*\}', text))
    if tags:
        print(f"  [KaTeX WARNING] \\tag found {len(tags)} times")
        for t in tags[:3]:
            print(f"    {t.group(0)}")

    bms = list(re.finditer(r'\\bm\{[^}]*\}', text))
    if bms:
        print(f"  [KaTeX WARNING] \\bm found {len(bms)} times")

    aligns = list(re.finditer(r'\\begin\{align\}', text))
    if aligns:
        print(f"  [KaTeX WARNING] \\begin{{align}} found {len(aligns)} times")

    # Check table pipe collisions
    table_lines_with_math_pipe = []
    for line_idx, line in enumerate(text.splitlines()):
        if '|' in line:
            # find all $...$ in line
            math_blocks = re.findall(r'\$([^$]+)\$', line)
            for mb in math_blocks:
                if '|' in mb:
                    table_lines_with_math_pipe.append((line_idx + 1, mb))
    if table_lines_with_math_pipe:
        print(f"  [KaTeX/Table WARNING] Pipe '|' inside inline math in table row found {len(table_lines_with_math_pipe)} times:")
        for l_num, mb in table_lines_with_math_pipe[:3]:
            print(f"    Line {l_num}: ${mb}$")
