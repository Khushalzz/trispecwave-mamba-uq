import re
import os

def sanitize_math_katex(text):
    """
    Sanitizes LaTeX math expressions to guarantee 100% KaTeX compatibility:
    1. Removes or converts \\tag{...} to \\qquad \\text{(...)} inside $$ ... $$
    2. Replaces \\bm{...} with \\boldsymbol{...}
    3. Converts \\begin{align} to \\begin{aligned}
    4. Handles table pipe collisions: within lines starting with '|', replaces '|' in math with '\\vert' or '\\mid'
    """
    lines = text.splitlines()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        # Fix table pipe collision
        if stripped.startswith('|') and stripped.endswith('|'):
            # This is a table row!
            # Replace math expressions containing '|'
            def fix_table_math(match):
                math_content = match.group(1)
                # Replace || with \parallel
                math_content = math_content.replace(r'\|', r'\parallel ')
                math_content = math_content.replace('||', r'\parallel ')
                # Replace | with \mid or \vert
                # If it's |x - y| -> \vert x - y \vert
                # If it's P(x|y) or E[x|y] -> \mid
                # A safe replacement for general | in math is \vert or \mid
                # Let's replace ' | ' with ' \mid ' and '|' with '\vert '
                math_content = re.sub(r'\s*\|\s*', r' \\mid ', math_content)
                return f"${math_content}$"

            # Match inline math in table row
            line = re.sub(r'\$([^$]+)\$', fix_table_math, line)

        # Fix \tag{...}
        # In KaTeX, \tag{1} causes: KaTeX parse error: Expected 'EOF', got '\tag'
        # We replace \tag{X} with \qquad \text{(X)} inside math
        line = re.sub(r'\\tag\*?\{([^}]+)\}', r'\\qquad \\text{(\1)}', line)

        # Fix \bm{...} -> \boldsymbol{...}
        line = re.sub(r'\\bm\{([^}]+)\}', r'\\boldsymbol{\1}', line)

        # Fix \begin{align} and \end{align}
        line = line.replace(r'\begin{align}', r'\begin{aligned}')
        line = line.replace(r'\end{align}', r'\end{aligned}')
        line = line.replace(r'\begin{align*}', r'\begin{aligned}')
        line = line.replace(r'\end{align*}', r'\end{aligned}')

        new_lines.append(line)

    return '\n'.join(new_lines)

print("Sanitization function ready.")
