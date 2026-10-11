# -*- coding: utf-8 -*-
"""
Converts docs/complete_paper_manuscript.md into a publication-grade Microsoft Word (.docx) document.
Features:
- IEEE / Professional Academic Paper typography (Times New Roman / Calibri / Consolas)
- Deep Navy and Slate Blue color palette for headings
- High-quality native Word tables with styled headers, alternating row shading, and cell borders
- Cleanly formatted display equations and inline math using Unicode mathematical typography
- Shaded, monospaced ASCII architecture boxes
- Professional margins, line spacing, and hanging indents for references
"""

import re
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def latex_to_unicode(text):
    """
    Converts LaTeX math expressions into clean Unicode mathematical text.
    """
    # Superscripts
    text = text.replace('^2', '²').replace('^3', '³').replace('^T', 'ᵀ').replace(r'^\top', 'ᵀ')
    text = text.replace('^{-4}', '⁻⁴').replace('^{-6}', '⁻⁶').replace(r'^\circ', '°')
    
    # Subscripts
    text = text.replace(r'_{10\text{m}}', '₁₀ₘ').replace(r'_{20\text{m}}', '₂₀ₘ').replace(r'_{60\text{m}}', '₆₀ₘ')
    text = text.replace(r'_{\text{fused}}', '_fused').replace(r'_{\text{gate}}', '_gate')
    text = text.replace(r'_{\text{spec}}', '_spec').replace(r'_{\text{up}}', '_up')
    text = text.replace(r'_{\text{hetero}}', '_hetero').replace(r'_{\text{MTF}}', '_MTF')
    text = text.replace(r'_{\text{SW}}', '_SW').replace(r'_{\text{SAM}}', '_SAM')
    text = text.replace(r'_{\text{NDVI}}', '_NDVI').replace(r'_{\text{NDRE}}', '_NDRE').replace(r'_{\text{NDWI}}', '_NDWI')
    text = text.replace('_{b,i}', '_b,i').replace('_{i,j}', '_i,j').replace('_{b-1}', '_b-1')
    
    # Greek letters
    greek = {
        r'\boldsymbol{\mu}': 'μ', r'\boldsymbol{\sigma}': 'σ',
        r'\mathbf{\mu}': 'μ', r'\mathbf{\sigma}': 'σ',
        r'\mu': 'μ', r'\sigma': 'σ', r'\lambda': 'λ', r'\Omega': 'Ω',
        r'\epsilon': 'ε', r'\nabla': '∇', r'\eta': 'η', r'\beta': 'β',
        r'\rho': 'ρ', r'\Delta': 'Δ', r'\theta': 'θ', r'\alpha': 'α',
    }
    for k, v in greek.items():
        text = text.replace(k, v)
        
    # Math operators and symbols
    symbols = {
        r'\rightarrow': '→', r'\leftarrow': '←', r'\times': '×',
        r'\le': '≤', r'\ge': '≥', r'\approx': '≈', r'\neq': '≠',
        r'\in': '∈', r'\notin': '∉', r'\sum': '∑', r'\prod': '∏',
        r'\partial': '∂', r'\infty': '∞', r'\pm': '±', r'\circ': '°',
        r'\Vert': '‖', r'\vert': '|', r'\mid': '|', r'\parallel': '‖',
        r'\ast': '∗', r'\cdot': '·', r'\dots': '...', r'\ldots': '...',
        r'\mathcal{O}': 'O', r'\mathbb{R}': 'ℝ', r'\mathbb{E}': 'E',
        r'\mathcal{I}': 'I', r'\mathcal{E}': 'E', r'\mathcal{D}': 'D',
        r'\mathcal{N}': 'N', r'\mathcal{F}': 'F',
        r'\,': ' ', r'\quad': '  ', r'\qquad': '    ',
        r'\uparrow': '↑', r'\downarrow': '↓',
        r'\left(': '(', r'\right)': ')',
        r'\left[': '[', r'\right]': ']',
        r'\{': '{', r'\}': '}',
        r'\left\Vert': '‖', r'\right\Vert': '‖',
        r'\left\vert': '|', r'\right\vert': '|',
        r'\log_10': 'log₁₀', r'\log_{10}': 'log₁₀',
        r'\log': 'log', r'\exp': 'exp',
        r'\arccos': 'arccos', r'\Softmax': 'Softmax',
        r'\text{Softmax}': 'Softmax',
    }
    for k, v in symbols.items():
        text = text.replace(k, v)
        
    # Formatting regexes
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\boldsymbol\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1) / (\2)', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', text)
    text = re.sub(r'\\text{(\d+)}', r'(\1)', text)

    return text.strip()

def add_styled_cell_border(cell, top="CCCCCC", bottom="CCCCCC", left=None, right=None):
    """Adds clean borders to table cells."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    borders = {'w:top': top, 'w:bottom': bottom, 'w:left': left, 'w:right': right}
    for border_name, color in borders.items():
        if color:
            b = OxmlElement(border_name)
            b.set(qn('w:val'), 'single')
            b.set(qn('w:sz'), '4')
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), color)
            tcBorders.append(b)
        else:
            b = OxmlElement(border_name)
            b.set(qn('w:val'), 'none')
            tcBorders.append(b)
    tcPr.append(tcBorders)

def add_cell_shading(cell, color_hex):
    """Sets background shading of a cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def add_formatted_text_to_para(p, text, base_font="Times New Roman", base_size=10.5, base_color=(51, 51, 51)):
    """
    Parses Markdown inline styling (**bold**, *italic*, `code`, $math$) and adds formatted runs.
    """
    pattern = re.compile(r'(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`|\$[^$]+?\$|<br\s*/?>)')
    tokens = pattern.split(text)

    for token in tokens:
        if not token:
            continue
        if token.lower() in ('<br>', '<br/>', '<br />'):
            p.add_run('\n')
            continue

        run = p.add_run()
        run.font.name = base_font
        run.font.size = Pt(base_size)
        run.font.color.rgb = RGBColor(*base_color)

        if token.startswith('**') and token.endswith('**'):
            inner = token[2:-2]
            if '$' in inner:
                inner = re.sub(r'\$([^$]+)\$', lambda m: latex_to_unicode(m.group(1)), inner)
            run.text = inner
            run.bold = True
        elif token.startswith('*') and token.endswith('*'):
            inner = token[1:-1]
            if '$' in inner:
                inner = re.sub(r'\$([^$]+)\$', lambda m: latex_to_unicode(m.group(1)), inner)
            run.text = inner
            run.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run.text = token[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 0.5)
            run.font.color.rgb = RGBColor(160, 40, 60)
        elif token.startswith('$') and token.endswith('$'):
            math_text = latex_to_unicode(token[1:-1])
            run.text = math_text
            run.font.name = "Cambria Math"
            run.font.size = Pt(base_size)
            run.italic = True
        else:
            run.text = token

def build_docx(md_path, docx_path):
    print(f"Reading markdown from {md_path}...")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = Document()

    # Set page margins: 0.85 inch for a clean academic journal feel
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    NAVY = (27, 54, 93)      # #1B365D Primary Heading
    SLATE = (44, 77, 117)    # #2C4D75 Secondary Heading
    DARK = (33, 37, 41)      # #212529 Body text
    MUTED = (108, 117, 125)  # #6C757D Muted / Subtitle
    CODE_BG = "F4F5F7"
    TABLE_HEADER_BG = "1B365D"
    ZEBRA_BG = "F8F9FA"

    lines = md_text.splitlines()
    i = 0
    total_lines = len(lines)

    in_code_block = False
    code_lines = []

    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        # Handle Code / ASCII Blocks
        if stripped.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_lines = []
                i += 1
                continue
            else:
                in_code_block = False
                # Render code block as shaded table cell for clean boxing
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = tbl.cell(0, 0)
                add_cell_shading(cell, CODE_BG)
                add_styled_cell_border(cell, top="CCCCCC", bottom="CCCCCC", left="CCCCCC", right="CCCCCC")
                cell_p = cell.paragraphs[0]
                cell_p.paragraph_format.space_before = Pt(4)
                cell_p.paragraph_format.space_after = Pt(4)
                cell_p.paragraph_format.line_spacing = 1.05
                run = cell_p.add_run('\n'.join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(30, 41, 59)
                
                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_before = Pt(0)
                spacer.paragraph_format.space_after = Pt(4)
                i += 1
                continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Blank lines
        if not stripped:
            i += 1
            continue

        # Horizontal rule ---
        if stripped == '---':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="E2E8F0"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)
            i += 1
            continue

        # Display Math $$ ... $$
        if stripped.startswith('$$'):
            math_lines = []
            if stripped.endswith('$$') and len(stripped) > 2:
                math_lines.append(stripped[2:-2].strip())
            else:
                i += 1
                while i < total_lines and not lines[i].strip().endswith('$$'):
                    math_lines.append(lines[i].strip())
                    i += 1
                if i < total_lines and lines[i].strip().endswith('$$'):
                    tail = lines[i].strip()[:-2].strip()
                    if tail:
                        math_lines.append(tail)

            raw_math = ' '.join(math_lines)
            unicode_math = latex_to_unicode(raw_math)

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            
            run = p.add_run(unicode_math)
            run.font.name = "Cambria Math"
            run.font.size = Pt(11)
            run.italic = True
            run.font.color.rgb = RGBColor(15, 23, 42)
            i += 1
            continue

        # Headings
        if stripped.startswith('# '):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(title_text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(20)
            run.bold = True
            run.font.color.rgb = RGBColor(*NAVY)
            i += 1
            continue

        if stripped.startswith('## '):
            h_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(13.5)
            run.bold = True
            run.font.color.rgb = RGBColor(*NAVY)
            i += 1
            continue

        if stripped.startswith('### '):
            h_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(11.5)
            run.bold = True
            run.font.color.rgb = RGBColor(*SLATE)
            i += 1
            continue

        if stripped.startswith('#### '):
            h_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(10.5)
            run.bold = True
            run.italic = True
            run.font.color.rgb = RGBColor(*SLATE)
            i += 1
            continue

        # Markdown Table Detection
        if stripped.startswith('|') and stripped.endswith('|'):
            table_lines = []
            while i < total_lines and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                rows_data = []
                for t_line in table_lines:
                    cells = [c.strip() for c in t_line.split('|')[1:-1]]
                    if all(re.match(r'^:?-+:?$', c) for c in cells if c):
                        continue
                    rows_data.append(cells)

                if rows_data:
                    num_cols = max(len(r) for r in rows_data)
                    tbl = doc.add_table(rows=len(rows_data), cols=num_cols)
                    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    tbl.autofit = True

                    for row_idx, row in enumerate(rows_data):
                        is_header = (row_idx == 0)
                        docx_row = tbl.rows[row_idx]
                        
                        trPr = docx_row._tr.get_or_add_trPr()
                        if is_header:
                            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

                        for col_idx in range(num_cols):
                            cell = docx_row.cells[col_idx]
                            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                            cell_text = row[col_idx] if col_idx < len(row) else ""

                            if is_header:
                                add_cell_shading(cell, TABLE_HEADER_BG)
                                add_styled_cell_border(cell, top="1B365D", bottom="1B365D", left="2C4D75", right="2C4D75")
                            else:
                                if row_idx % 2 == 1:
                                    add_cell_shading(cell, ZEBRA_BG)
                                add_styled_cell_border(cell, top="E2E8F0", bottom="E2E8F0", left="E2E8F0", right="E2E8F0")

                            cell_p = cell.paragraphs[0]
                            cell_p.paragraph_format.space_before = Pt(3)
                            cell_p.paragraph_format.space_after = Pt(3)
                            cell_p.paragraph_format.line_spacing = 1.05

                            if is_header:
                                add_formatted_text_to_para(cell_p, cell_text, base_font="Times New Roman", base_size=9.5, base_color=(255, 255, 255))
                                for r in cell_p.runs:
                                    r.bold = True
                            else:
                                add_formatted_text_to_para(cell_p, cell_text, base_font="Times New Roman", base_size=9.0, base_color=DARK)

                    post_tbl_p = doc.add_paragraph()
                    post_tbl_p.paragraph_format.space_before = Pt(0)
                    post_tbl_p.paragraph_format.space_after = Pt(6)
            continue

        # Bullet List Items
        if stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text_to_para(p, bullet_text, base_font="Times New Roman", base_size=10.5, base_color=DARK)
            i += 1
            continue

        # Numbered List Items
        num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if num_match:
            num_val = num_match.group(1)
            item_text = num_match.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.left_indent = Inches(0.25)
            r_num = p.add_run(f"{num_val}. ")
            r_num.font.name = "Times New Roman"
            r_num.font.size = Pt(10.5)
            r_num.bold = True
            r_num.font.color.rgb = RGBColor(*DARK)
            add_formatted_text_to_para(p, item_text, base_font="Times New Roman", base_size=10.5, base_color=DARK)
            i += 1
            continue

        # Standard Text Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15

        if stripped.startswith('**Khushalzz et al.**') or stripped.startswith('*Laboratory for Geospatial'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_formatted_text_to_para(p, stripped, base_font="Times New Roman", base_size=11, base_color=MUTED)
        elif stripped.startswith('**Keywords:**'):
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            add_formatted_text_to_para(p, stripped, base_font="Times New Roman", base_size=10, base_color=DARK)
        else:
            add_formatted_text_to_para(p, stripped, base_font="Times New Roman", base_size=10.5, base_color=DARK)

        i += 1

    print(f"Saving formatted Word document to {docx_path}...")
    doc.save(docx_path)
    print("Done!")

if __name__ == '__main__':
    md_file = 'docs/complete_paper_manuscript.md'
    docx_file = 'docs/complete_paper_manuscript.docx'
    docx_named = 'docs/TriSpecWaveMambaUQ_Paper.docx'
    build_docx(md_file, docx_file)
    build_docx(md_file, docx_named)
    print(f"Generated {docx_file} and {docx_named} successfully!")
