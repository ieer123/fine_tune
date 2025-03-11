import re
import unicodedata

def clean_markdown_math(content):
    # 扩展希腊字母和符号映射
    greek_symbols = {
        r'\\alpha': 'α', r'$\\alpha$': 'α', r'$\alpha$': 'α',
        r'\\beta': 'β', r'$\\beta$': 'β', r'$\beta$': 'β',
        r'\\gamma': 'γ', r'$\\gamma$': 'γ', r'$\gamma$': 'γ',
        r'\\Gamma': 'Γ', r'$\\Gamma$': 'Γ', r'$\Gamma$': 'Γ',
        r'\\delta': 'δ', r'$\\delta$': 'δ', r'$\delta$': 'δ',
        r'\\Delta': 'Δ', r'$\\Delta$': 'Δ', r'$\Delta$': 'Δ',
        r'\\cdot': '·', r'$\\cdot$': '·', r'$\cdot$': '·',
        r'\\times': '×', r'$\\times$': '×', r'$\times$': '×',
        r'\\mu': 'μ', r'$\\mu$': 'μ', r'$\mu$': 'μ',
        r'\\textmu': 'μ',
        r'\\circ': '°', r'$\\circ$': '°', r'$\circ$': '°',
    }

    # 处理内联公式的函数
    def process_inline_formula(formula):
        result = formula
        for pattern, replacement in patterns:
            try:
                result = re.sub(pattern, replacement, result, flags=re.DOTALL)
            except re.error:
                continue
        return result

    # 定义替换规则列表
    patterns = [
        (r'\$\$(.*?)\$\$', lambda m: process_inline_formula(m.group(1))),
        (r'\$(.*?)\$', lambda m: process_inline_formula(m.group(1))),
        *[(re.escape(k), v) for k, v in greek_symbols.items()],
    ]
    
    # 处理公式
    for pattern, replacement in patterns:
        try:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        except re.error:
            continue
    
    return content

def remove_redundant_phrases(text):
    # 删除冗余表述
    redundant_phrases = [
        r"在此不再展开",
        r"需强调的一点：",
        r"（.*?）",
        r"需要说明的是",
        r"值得注意的是"
    ]
    for phrase in redundant_phrases:
        text = re.sub(phrase, '', text)
    return text

def normalize_punctuation(text):
    # 增强的标点标准化
    replacements = [
        (r'[。，；：]\s*', lambda m: m.group()[0] + ' '),  # 中文标点后加空格
        (r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2'),       # 中英文间加空格
        (r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2'),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return unicodedata.normalize('NFKC', text)

def process_chemical_formulas(text):
    """处理文本中的化学式，将下标转换为Unicode下标"""
    # 处理形如H2O的常见化学式
    text = re.sub(r'([A-Z][a-z]*)(\d+)', r'\1₍\2₎', text)
    # 将括号内的数字转换为下标
    text = re.sub(r'₍(\d+)₎', lambda m: ''.join(chr(0x2080 + int(c)) for c in m.group(1)), text)
    return text

def process_section(text):
    """处理单个文本段落的辅助函数"""
    if not text.strip():
        return ""
        
    processed = text
    processed = re.sub(r'#{1,6}\s*', '', processed)          # 删除标题标记
    processed = clean_markdown_math(processed)                # 数学公式处理
    processed = process_chemical_formulas(processed)          # 化学式处理
    processed = remove_redundant_phrases(processed)           # 语义清理
    processed = normalize_punctuation(processed)              # 标点标准化

    # 清理残留的LaTeX标记
    processed = re.sub(r'\\[a-zA-Z]+(\{[^\}]*\})*', '', processed)
    processed = re.sub(r'\{|\}', '', processed)              # 清理残留的花括号
    
    # 基础文本清理，但保持段落格式
    processed = re.sub(r'\s+', ' ', processed)               # 合并多个空格
    processed = processed.strip()
    
    return processed

def process_markdown(input_path, output_path):
    """处理Markdown文件，保持原有的段落结构"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 保存代码块和图片引用
    code_blocks = []
    image_refs = []
    
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"CODE_BLOCK_{len(code_blocks)-1}"
        
    def save_image_ref(match):
        image_refs.append(match.group(0))
        return f"IMAGE_REF_{len(image_refs)-1}"

    # 保存特殊内容
    processed = content
    processed = re.sub(r'`{3}.*?`{3}', save_code_block, processed, flags=re.DOTALL)
    processed = re.sub(r'!\[.*?\]\(.*?\)', save_image_ref, processed, flags=re.DOTALL)
    
    # 按原始段落分割文本
    paragraphs = []
    current_paragraph = []
    
    for line in processed.split('\n'):
        if line.strip().startswith(('CODE_BLOCK_', 'IMAGE_REF_')):
            # 处理并保存当前段落
            if current_paragraph:
                processed_text = process_section('\n'.join(current_paragraph))
                if processed_text:
                    paragraphs.append(processed_text)
                current_paragraph = []
            
            # 还原特殊内容
            if line.strip().startswith('CODE_BLOCK_'):
                index = int(line.strip().split('_')[2])
                paragraphs.append(code_blocks[index])
            elif line.strip().startswith('IMAGE_REF_'):
                index = int(line.strip().split('_')[2])
                paragraphs.append(image_refs[index])
        else:
            if not line.strip() and current_paragraph:
                # 空行标志着段落的结束
                processed_text = process_section('\n'.join(current_paragraph))
                if processed_text:
                    paragraphs.append(processed_text)
                current_paragraph = []
            elif line.strip():
                current_paragraph.append(line)

    # 处理最后一个段落
    if current_paragraph:
        processed_text = process_section('\n'.join(current_paragraph))
        if processed_text:
            paragraphs.append(processed_text)

    # 使用双换行符连接段落，保持原有的段落分隔
    final_text = '\n\n'.join(paragraphs)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

if __name__ == '__main__':
    # 使用示例
    process_markdown('E:/code/disst/pdf/mterial_/mterial_.md', 'material_text.txt') 