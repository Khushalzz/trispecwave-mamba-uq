import re
from check_katex import check_file_math

path = 'docs/complete_paper_manuscript.md'
issues = check_file_math(path)
print(f"Total issues found: {len(issues)}")
for l_num, desc, snip in issues:
    # Filter out false positives like \rightarrow
    if "Mismatched \left" in desc and r"\rightarrow" in snip and r"\left" not in snip:
        continue
    print(f"  Line {l_num}: {desc} -> {snip[:80]}")

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

forbidden = ['sih', 'smart india hackathon', 'ps-2', 'ps2', 'problem statement']
for w in forbidden:
    matches = list(re.finditer(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE))
    print(f"Forbidden word '{w}': {len(matches)} matches")

# Check equation numbers or tag residual
tags = list(re.finditer(r'\\tag', text))
print(f"Residual \\tag count: {len(tags)}")

# Check unmatched $$
dollar_count = text.count('$$')
print(f"$$ delimiter count: {dollar_count} (even={dollar_count % 2 == 0})")
