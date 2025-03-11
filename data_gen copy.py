# -*- coding: utf-8 -*-

def process_text(text):
    # Flag to track if we're inside curly braces
    inside_braces = False
    result = []
    
    for i, char in enumerate(text):
        if char == '{':
            inside_braces = True
            result.append(char)
        elif char == '}':
            inside_braces = False
            result.append(char)
            # Add comma after closing brace if it's not already there
            if i + 1 < len(text) and text[i + 1] != ',':
                result.append(',')
        elif char == '，' and inside_braces:  # Chinese comma
            result.append(',')  # Replace with English comma
        else:
            result.append(char)
    
    return ''.join(result)

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    processed_content = process_text(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_content)

if __name__ == '__main__':
    # Example usage
    input_file = './api/instruction_dataset.json'  # Replace with your input file path
    output_file = './api/instruction_dataset_processed.json'  # Replace with your output file path
    process_file(input_file, output_file) 