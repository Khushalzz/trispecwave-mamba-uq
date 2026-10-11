# -*- coding: utf-8 -*-
"""
TriSpecWaveMambaUQ AIES Paper → Word (.docx) Converter
Completely rewritten for clean, professional academic layout.
"""

import re
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

# ─── COLOR PALETTE ───────────────────────────────────────────────────────────
NAVY   = (15,  44,  95)   # Deep navy — section headings
BLUE   = (25,  82,  156)  # Medium blue — subsection headings
SLATE  = (55,  85,  120)  # Slate — sub-subsection headings
DARK   = (20,  20,  30)   # Near-black body text
MUTED  = (90,  90,  110)  # Muted grey — captions, metadata
WHITE  = (255, 255, 255)
HEADER_BG = "1B365D"      # Dark navy table headers
ZEBRA_BG  = "EEF2F8"      # Light blue alternating rows


# ─── LATEX → UNICODE (for Word display) ──────────────────────────────────────
def clean_math(text):
    """Convert LaTeX math into readable Unicode for Word paragraphs."""
    # Strip $...$ inline math delimiters
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', r'\1', text)

    replacements = [
        # Greek
        (r'\boldsymbol{\mu}', 'μ'), (r'\boldsymbol{\sigma}', 'σ'),
        (r'\mathbf{\mu}', 'μ'),    (r'\mathbf{\sigma}', 'σ'),
        (r'\boldsymbol{', ''),      (r'\mathbf{', ''),
        (r'\mathcal{O}', 'O'),      (r'\mathcal{L}', 'ℒ'),
        (r'\mathcal{I}', 'I'),      (r'\mathcal{D}', 'D'),
        (r'\mathcal{E}', 'E'),      (r'\mathcal{N}', 'N'),
        (r'\mathcal{F}', 'F'),
        (r'\mathbb{E}', 'E'),       (r'\mathbb{R}', 'ℝ'),
        (r'\mu', 'μ'),  (r'\sigma', 'σ'),  (r'\lambda', 'λ'),
        (r'\Omega', 'Ω'), (r'\epsilon', 'ε'), (r'\nabla', '∇'),
        (r'\eta', 'η'),  (r'\beta', 'β'),   (r'\rho', 'ρ'),
        (r'\Delta', 'Δ'), (r'\theta', 'θ'), (r'\alpha', 'α'),
        # Operators
        (r'\rightarrow', '→'), (r'\leftarrow', '←'), (r'\times', '×'),
        (r'\le', '≤'), (r'\ge', '≥'), (r'\approx', '≈'), (r'\neq', '≠'),
        (r'\in', '∈'),  (r'\notin', '∉'),
        (r'\sum', 'Σ'), (r'\prod', 'Π'), (r'\partial', '∂'), (r'\infty', '∞'),
        (r'\pm', '±'),  (r'\cdot', '·'), (r'\dots', '…'), (r'\ldots', '…'),
        (r'\uparrow', '↑'), (r'\downarrow', '↓'),
        (r'\Vert', '‖'),  (r'\vert', '|'), (r'\mid', '|'),
        (r'\left(', '('), (r'\right)', ')'),
        (r'\left[', '['), (r'\right]', ']'),
        (r'\left\{', '{'), (r'\right\}', '}'),
        (r'\left\Vert', '‖'), (r'\right\Vert', '‖'),
        (r'\left\vert', '|'), (r'\right\vert', '|'),
        (r'\log_{10}', 'log₁₀'), (r'\log', 'log'), (r'\exp', 'exp'),
        (r'\arccos', 'arccos'), (r'\text{Softmax}', 'Softmax'),
        (r'\top', 'ᵀ'), (r'\circ', '°'),
        (r'\quad', '  '), (r'\qquad', '    '), (r'\,', ' '),
        (r'\\', ' '),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    # \frac{a}{b} → (a)/(b)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    # \sqrt{a} → √(a)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', text)
    # \text{...} → plain
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    # ^{2} → ² style superscripts
    text = text.replace('^2', '²').replace('^3', '³').replace('^T', 'ᵀ')
    text = re.sub(r'\^\{([^}]+)\}', r'^\1', text)  # keep remaining as ^x
    # _{x} → subscript notation
    text = re.sub(r'_\{([^}]+)\}', r'_\1', text)
    # Strip remaining braces
    text = re.sub(r'[\{\}]', '', text)

    return text.strip()


# ─── XML HELPERS ─────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_border(cell, color="CCCCCC"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side in ['w:top', 'w:left', 'w:bottom', 'w:right']:
        b = OxmlElement(side)
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), '4')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        tcBorders.append(b)
    tcPr.append(tcBorders)


def set_table_border(tbl, color="AAAAAA"):
    tblPr = tbl._tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl._tbl.insert(0, tblPr)
    tblBorders = OxmlElement('w:tblBorders')
    for side in ['w:top', 'w:left', 'w:bottom', 'w:right', 'w:insideH', 'w:insideV']:
        b = OxmlElement(side)
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), '4')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        tblBorders.append(b)
    tblPr.append(tblBorders)


def set_run_font(run, name="Times New Roman", size_pt=10.5, bold=False,
                 italic=False, color_rgb=None):
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = RGBColor(*color_rgb)


# ─── INLINE TEXT RENDERER ────────────────────────────────────────────────────
def add_inline_text(para, text, base_size=10.5, base_color=DARK,
                    base_font="Times New Roman", base_bold=False,
                    base_italic=False):
    """
    Render markdown inline formatting: **bold**, *italic*, `code`, inline $math$.
    """
    # Pre-process: replace inline $...$ with [MATH:...:MATH]
    text = re.sub(r'\$([^$\n]+?)\$', lambda m: '[MATH:' + m.group(1) + ':MATH]', text)

    # Tokenise on **bold**, *italic*, `code`, [MATH:...:MATH]
    pattern = re.compile(
        r'(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`|\[MATH:.+?:MATH\])'
    )
    segments = pattern.split(text)

    for seg in segments:
        if not seg:
            continue
        seg_clean = clean_math(seg)

        if seg.startswith('**') and seg.endswith('**'):
            inner = clean_math(seg[2:-2])
            run = para.add_run(inner)
            set_run_font(run, base_font, base_size, bold=True, color_rgb=base_color)
        elif seg.startswith('*') and seg.endswith('*') and not seg.startswith('**'):
            inner = clean_math(seg[1:-1])
            run = para.add_run(inner)
            set_run_font(run, base_font, base_size, italic=True, color_rgb=base_color)
        elif seg.startswith('`') and seg.endswith('`'):
            inner = seg[1:-1]
            run = para.add_run(inner)
            set_run_font(run, "Consolas", base_size - 0.5, color_rgb=(100, 50, 160))
        elif seg.startswith('[MATH:') and seg.endswith(':MATH]'):
            math_src = seg[6:-6]
            rendered = clean_math(math_src)
            run = para.add_run(rendered)
            set_run_font(run, "Cambria Math", base_size, italic=True,
                         color_rgb=(15, 40, 100))
        else:
            run = para.add_run(clean_math(seg))
            set_run_font(run, base_font, base_size, bold=base_bold,
                         italic=base_italic, color_rgb=base_color)


# ─── DISPLAY MATH BLOCK ──────────────────────────────────────────────────────
def add_display_math(doc, raw_latex):
    rendered = clean_math(raw_latex)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.right_indent = Inches(0.5)
    run = p.add_run(rendered)
    set_run_font(run, "Cambria Math", 11, italic=True, color_rgb=(10, 30, 80))
    return p


# ─── PARAGRAPH HELPERS ───────────────────────────────────────────────────────
def add_body_para(doc, text, indent=0, space_before=2, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.2
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    add_inline_text(p, text)
    return p


def add_heading(doc, text, level=1):
    """Level 1 = Section (##), Level 2 = Subsection (###), Level 3 = Sub-sub (####)."""
    p = doc.add_paragraph()

    if level == 0:
        # Paper title
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(10)
        run = p.add_run(clean_math(text))
        set_run_font(run, "Times New Roman", 16, bold=True, color_rgb=NAVY)

    elif level == 1:
        # Major section: ## I. Introduction
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(clean_math(text).upper())
        set_run_font(run, "Times New Roman", 11.5, bold=True, color_rgb=NAVY)
        # Add a bottom border line
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), '1B365D')
        pBdr.append(bottom)
        pPr.append(pBdr)

    elif level == 2:
        # Subsection: ### A. Whatever
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(clean_math(text))
        set_run_font(run, "Times New Roman", 11, bold=True, italic=True,
                     color_rgb=BLUE)

    elif level == 3:
        # Sub-subsection: #### something
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(clean_math(text))
        set_run_font(run, "Times New Roman", 10.5, bold=True, color_rgb=SLATE)

    return p


def add_figure(doc, img_path, caption, repo_dir):
    """Embed a figure with a centered caption below it."""
    candidates = [
        img_path,
        os.path.join(repo_dir, img_path),
        os.path.join(os.getcwd(), img_path),
    ]
    resolved = None
    for c in candidates:
        if os.path.exists(c):
            resolved = c
            break

    if resolved:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(14)
        p_img.paragraph_format.space_after = Pt(4)
        p_img.paragraph_format.keep_with_next = True
        run = p_img.add_run()
        # Scale to page width (max 5.8 inches for the text area)
        run.add_picture(resolved, width=Inches(5.8))
    else:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p_img.add_run(f"[Figure not found: {img_path}]")
        set_run_font(run, "Times New Roman", 10, italic=True, color_rgb=(200, 50, 50))

    # Caption paragraph
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(14)
    add_inline_text(p_cap, caption, base_size=9.5, base_color=MUTED,
                    base_italic=True)


def add_table_from_rows(doc, rows):
    """
    Render a list of row-lists into a proper Word table with:
    - Dark navy header row with white text
    - Alternating row shading
    - Full borders on all cells
    """
    if not rows:
        return

    n_cols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = 'Table Grid'

    for r_idx, row_data in enumerate(rows):
        is_header = (r_idx == 0)
        docx_row = tbl.rows[r_idx]
        docx_row.height = Pt(18) if is_header else Pt(15)

        for c_idx in range(n_cols):
            cell = docx_row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell_text = row_data[c_idx] if c_idx < len(row_data) else ""

            if is_header:
                set_cell_bg(cell, HEADER_BG)
                set_cell_border(cell, "2C4D75")
            else:
                if r_idx % 2 == 0:
                    set_cell_bg(cell, ZEBRA_BG)
                else:
                    set_cell_bg(cell, "FFFFFF")
                set_cell_border(cell, "C8D5E5")

            cell_p = cell.paragraphs[0]
            cell_p.paragraph_format.space_before = Pt(2)
            cell_p.paragraph_format.space_after = Pt(2)
            cell_p.paragraph_format.line_spacing = 1.1

            if is_header:
                add_inline_text(cell_p, cell_text, base_size=9,
                                base_color=WHITE, base_bold=True)
            else:
                add_inline_text(cell_p, cell_text, base_size=9,
                                base_color=DARK)

    # Spacing after table
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(8)


def add_hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_code_block(doc, lines_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.right_indent = Inches(0.3)
    # Grey background via paragraph border shading hack
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F0F0F0')
    pPr.append(shd)
    run = p.add_run(lines_text)
    set_run_font(run, "Consolas", 8.5, color_rgb=(30, 30, 30))


# ─── MAIN BUILDER ────────────────────────────────────────────────────────────
def build_docx(md_path, docx_path):
    print(f"Building {docx_path} from {md_path}...")

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(md_path)))

    with open(md_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    lines = raw.splitlines()
    doc = Document()

    # ── Page Setup ────────────────────────────────────────────────────────────
    for section in doc.sections:
        section.page_width  = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    # ── Default Style ─────────────────────────────────────────────────────────
    style = doc.styles['Normal']
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.font.color.rgb = RGBColor(*DARK)

    # ── Parse & Render ────────────────────────────────────────────────────────
    i = 0
    total = len(lines)

    # state
    in_code = False
    code_buf = []
    pending_table_header = None  # caption string for next table

    while i < total:
        line = lines[i]
        stripped = line.strip()

        # ── Skip blank lines (add tiny spacing instead) ───────────────────────
        if stripped == '':
            i += 1
            continue

        # ── Horizontal rule --- ───────────────────────────────────────────────
        if stripped in ('---', '***', '___'):
            add_hr(doc)
            i += 1
            continue

        # ── Fenced code block ````...``` ─────────────────────────────────────
        if stripped.startswith('```'):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                add_code_block(doc, '\n'.join(code_buf))
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # ── Figure image: ![...](path) ────────────────────────────────────────
        img_match = re.match(r'^!\[(.+?)\]\((.+?)\)\s*$', stripped)
        if img_match:
            img_caption_raw = img_match.group(1)
            img_path = img_match.group(2)
            # Try to grab the bold caption line immediately following
            caption_text = img_caption_raw
            if i + 1 < total:
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith('**Figure'):
                    caption_text = next_stripped
                    i += 1  # consume caption line too
            add_figure(doc, img_path, caption_text, repo_dir)
            i += 1
            continue

        # ── Bold figure caption (standalone) already consumed above, skip ─────
        if stripped.startswith('**Figure') and stripped.endswith('*'):
            # Orphaned caption — render as centered italic
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(12)
            add_inline_text(p, stripped, base_size=9.5, base_color=MUTED, base_italic=True)
            i += 1
            continue

        # ── Display math: $$ ... $$ ───────────────────────────────────────────
        if stripped.startswith('$$'):
            math_lines = []
            if stripped.endswith('$$') and len(stripped) > 4:
                math_lines.append(stripped[2:-2].strip())
            else:
                i += 1
                while i < total and not lines[i].strip().endswith('$$'):
                    math_lines.append(lines[i].strip())
                    i += 1
                if i < total:
                    tail = lines[i].strip()[:-2].strip()
                    if tail:
                        math_lines.append(tail)
            add_display_math(doc, ' '.join(math_lines))
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        if stripped.startswith('#### '):
            add_heading(doc, stripped[5:].strip(), level=3)
            i += 1
            continue

        if stripped.startswith('### '):
            add_heading(doc, stripped[4:].strip(), level=2)
            i += 1
            continue

        if stripped.startswith('## '):
            add_heading(doc, stripped[3:].strip(), level=1)
            i += 1
            continue

        if stripped.startswith('# '):
            add_heading(doc, stripped[2:].strip(), level=0)
            i += 1
            continue

        # ── Markdown table ────────────────────────────────────────────────────
        if stripped.startswith('|') and stripped.endswith('|'):
            table_raw = []
            while i < total and lines[i].strip().startswith('|'):
                table_raw.append(lines[i].strip())
                i += 1

            rows_data = []
            for trow in table_raw:
                cells = [c.strip() for c in trow.split('|')[1:-1]]
                # Skip separator rows (---, :---:, etc.)
                if all(re.match(r'^:?-{2,}:?$', c.replace(' ', '')) for c in cells if c):
                    continue
                rows_data.append(cells)

            # Check for table caption above (### Table...) — already rendered as heading
            # Render the table
            if pending_table_header:
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(4)
                p_cap.paragraph_format.space_after = Pt(3)
                add_inline_text(p_cap, pending_table_header, base_size=9.5,
                                base_color=BLUE, base_bold=True)
                pending_table_header = None

            add_table_from_rows(doc, rows_data)
            continue  # i already advanced in the while loop

        # ── Table caption: "### Table X: ..." ────────────────────────────────
        # (detected as heading level 2, handled — nothing extra needed here)

        # ── Bullet list: - ... ────────────────────────────────────────────────
        if stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.2
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.2)
            run_bullet = p.add_run("• ")
            set_run_font(run_bullet, "Times New Roman", 10.5,
                         bold=True, color_rgb=BLUE)
            add_inline_text(p, bullet_text, base_size=10.5, base_color=DARK)
            i += 1
            continue

        # ── Numbered list: 1. ... ─────────────────────────────────────────────
        num_m = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if num_m:
            num = num_m.group(1)
            item_text = num_m.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.2
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.25)
            run_num = p.add_run(f"{num}. ")
            set_run_font(run_num, "Times New Roman", 10.5,
                         bold=True, color_rgb=NAVY)
            add_inline_text(p, item_text, base_size=10.5, base_color=DARK)
            i += 1
            continue

        # ── Indented continuation line (starts with spaces) ───────────────────
        indent_m = re.match(r'^( {3,})(.*)', line)
        if indent_m and not stripped.startswith('|'):
            indent_level = len(indent_m.group(1)) // 3
            content = indent_m.group(2).strip()
            if content.startswith('- ') or content.startswith('* '):
                # Nested bullet
                nested = content[2:].strip()
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.left_indent = Inches(0.5 + indent_level * 0.15)
                p.paragraph_format.first_line_indent = Inches(-0.2)
                run_b = p.add_run("◦ ")
                set_run_font(run_b, "Times New Roman", 10.5, color_rgb=SLATE)
                add_inline_text(p, nested, base_size=10.5, base_color=DARK)
            elif content.startswith('$$'):
                # Indented display math
                math_src = content.strip('$').strip()
                add_display_math(doc, math_src)
            else:
                # Continuation / italic intuition note
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.2
                p.paragraph_format.left_indent = Inches(0.4)
                add_inline_text(p, content, base_size=10.0, base_color=SLATE,
                                base_italic=True)
            i += 1
            continue

        # ── Title metadata block (AIES MINI PROJECT, Authors, etc.) ──────────
        if any(kw in stripped for kw in ['AIES MINI PROJECT', 'Department of Computer',
                                          'Authors:', 'Academic Year:', 'Domain:']):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_inline_text(p, stripped, base_size=10, base_color=MUTED)
            i += 1
            continue

        # ── Keywords line ────────────────────────────────────────────────────
        if stripped.startswith('**Keywords:**') or stripped.startswith('**Key'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(10)
            add_inline_text(p, stripped, base_size=10, base_color=DARK)
            i += 1
            continue

        # ── Abstract heading ###Abstract ─────────────────────────────────────
        if stripped.lower() in ('### abstract', '**abstract**', 'abstract'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run("ABSTRACT")
            set_run_font(run, "Times New Roman", 11, bold=True, color_rgb=NAVY)
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bot = OxmlElement('w:bottom')
            bot.set(qn('w:val'), 'single')
            bot.set(qn('w:sz'), '4')
            bot.set(qn('w:space'), '1')
            bot.set(qn('w:color'), '1B365D')
            pBdr.append(bot)
            pPr.append(pBdr)
            i += 1
            continue

        # ── General body paragraph ────────────────────────────────────────────
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline_text(p, stripped, base_size=10.5, base_color=DARK)
        i += 1

    print(f"  Saving to {docx_path}...")
    doc.save(docx_path)
    print(f"  Done! ({os.path.getsize(docx_path) // (1024*1024)} MB)")


# ─── ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    MD   = 'docs/complete_paper_manuscript.md'
    OUT1 = 'docs/complete_paper_manuscript.docx'
    OUT2 = 'docs/TriSpecWaveMambaUQ_Paper.docx'

    build_docx(MD, OUT1)
    build_docx(MD, OUT2)
    print(f"\nGenerated:\n  -> {OUT1}\n  -> {OUT2}")
