import os
import json

# Variables to control script behavior
N = 10000  # Number of fine-tuning examples to generate
INCLUDE_COMMENTS = False  # Set to False to exclude comments from descriptions
BASE_DIR = 'oeisdata/seq'  # Base directory containing .seq files

def main():
    examples_generated = 0

    # Determine output file name based on INCLUDE_COMMENTS
    output_file = 'training_data.jsonl'
    if not INCLUDE_COMMENTS:
        output_file = 'training_data_no_comments.jsonl'

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, _ in os.walk(BASE_DIR):
            dirs.sort()  # Ensure consistent order
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if os.path.isdir(dir_path):
                    seq_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.seq')])
                    for file_name in seq_files:
                        file_path = os.path.join(dir_path, file_name)
                        example = process_seq_file(file_path)
                        if example:
                            out_f.write(json.dumps(example, ensure_ascii=False) + '\n')
                            examples_generated += 1
                            if examples_generated >= N:
                                break
                    if examples_generated >= N:
                        break
            if examples_generated >= N:
                break

    # Optionally, inspect the generated data
    inspect_data(output_file)

def process_seq_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sequence_lines = []
    title_lines = []
    comment_lines = []
    has_elements = False
    has_title = False

    for line in lines:
        line = line.strip()
        if line.startswith('%'):
            prefix = line[1]
            content = line[2:].strip()
            if prefix in ['S', 'T', 'U']:
                # Collect sequence elements
                parts = content.split(' ', 1)
                if len(parts) > 1:
                    sequence_lines.append(parts[1])
                else:
                    sequence_lines.append(parts[0])
                has_elements = True
            elif prefix == 'N':
                # Collect title lines
                title_lines.append(content)
                has_title = True
            elif prefix == 'C' and INCLUDE_COMMENTS:
                # Collect comment lines if including comments
                comment_lines.append(content)

    if not (has_elements and has_title):
        return None

    # Process sequence elements
    elements_str = ''.join(sequence_lines)
    elements_str = elements_str.split(' ', 1)[-1] if ' ' in elements_str else elements_str
    elements_str = elements_str.replace(' ', '').replace('\n', '').replace('\r', '')
    elements = elements_str.split(',')
    elements = [e for e in elements if e.strip().lstrip('-').isdigit()]

    if len(elements) < 2:
        return None  # Not enough terms to split

    # Split sequence into two halves
    split_index = len(elements) // 2
    first_half = elements[:split_index]
    second_half = elements[split_index:]

    # Prepare messages
    user_message = f"[BEGINNING SEQ]\nSequence: {', '.join(first_half)}\n[END SEQ]"
    description_parts = title_lines
    if INCLUDE_COMMENTS:
        description_parts += comment_lines
    description = '\n'.join(description_parts).strip()

    # Truncate the description if it exceeds 1000 characters
    if len(description) > 1000:
        description = description[:1000] + '...'

    assistant_message = f"[DESCRIPTION]\n{description}\n[END DESCRIPTION]\n[Remaining sequence terms: {', '.join(second_half)}]"

    # Create example
    example = {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that, given the beginning of an integer sequence, predicts the description and the next terms."
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_message
            }
        ]
    }

    return example

def inspect_data(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Total examples: {len(data)}\n")
    for i, example in enumerate(data[:3], 1):
        print(f"Example {i}:\n{json.dumps(example, indent=2, ensure_ascii=False)}\n")

if __name__ == '__main__':
    main()
