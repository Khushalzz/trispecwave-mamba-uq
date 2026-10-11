import re

def check_file_math(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.splitlines()
    in_display_math = False
    display_math_content = []
    display_start = 0

    issues = []

    for i, line in enumerate(lines):
        line_num = i + 1

        # Check table line
        is_table = line.strip().startswith('|') and line.strip().endswith('|')
        if is_table:
            # check if any unescaped pipe inside $...$
            matches = re.findall(r'\$([^$]+)\$', line)
            for m in matches:
                if '|' in m:
                    issues.append((line_num, "Raw pipe '|' inside inline math in table row", m))
                if r'\|' in m:
                    issues.append((line_num, r"Escaped pipe '\|' inside inline math in table row (may split Markdown table columns)", m))

        # Check $$ ... $$
        if '$$' in line:
            parts = line.split('$$')
            # toggle display math
            count = line.count('$$')
            if count % 2 == 1:
                in_display_math = not in_display_math

        if r'\tag{' in line or r'\tag*{' in line:
            issues.append((line_num, r"\tag found (causes KaTeX ParseError)", line))

        if r'\bm{' in line:
            issues.append((line_num, r"\bm found (causes KaTeX error if amsbsy not loaded)", line))

        # check for unmatched braces in $...$
        inline_matches = re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', line)
        for m in inline_matches:
            if m.count('{') != m.count('}'):
                issues.append((line_num, "Mismatched braces in inline math", m))

    return issues

if __name__ == '__main__':
    files = [
        'docs/section_ii_literature_survey.md',
        'docs/section_iii_methodology.md',
        'docs/section_iv_results_and_discussion.md',
        'docs/section_v_research_gaps_and_future_horizons.md',
        'docs/complete_paper_manuscript.md'
    ]

    for f in files:
        issues = check_file_math(f)
        print(f"File: {f} -> {len(issues)} issues found")
        for l_num, desc, snippet in issues[:5]:
            print(f"  Line {l_num}: {desc} -> {snippet[:60]}...")
