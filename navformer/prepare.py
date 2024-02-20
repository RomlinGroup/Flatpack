import numpy as np
import os
import re
import tiktoken

train_ids = []
val_ids = []

enc = tiktoken.get_encoding("gpt2")


def split_file(filename, output_dir, chunk_size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(filename, 'r') as f:
        lines = f.readlines()

    n_chunks = max(1, (len(lines) + chunk_size - 1) // chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(lines))
        chunk_lines = lines[start:end]
        output_filename = os.path.join(output_dir, f'{i}-dataset.txt')
        with open(output_filename, 'w') as f:
            f.writelines(chunk_lines)


split_file('dataset.txt', 'nanoChatGPT/output', 1000)


def get_file_number(filename):
    match = re.match(r"(\d+)-dataset\.txt", filename)
    return int(match.group(1)) if match else None


output_files = os.listdir('nanoChatGPT/output')
output_files.sort()

train_proportion = 0.8
n_files = len(output_files)
train_file_limit = int(n_files * train_proportion)

print(f"Total files: {n_files}, Training file limit: {train_file_limit}")

for filename in output_files:
    if filename.endswith('.txt'):
        file_number = get_file_number(filename)
        if file_number is not None:
            try:
                with open(f'nanoChatGPT/output/{filename}', 'r') as f:
                    data = f.read()
                if file_number < train_file_limit:
                    train_ids.extend(enc.encode_ordinary(data))
                    print(f"Adding to train: {filename}")
                else:
                    val_ids.extend(enc.encode_ordinary(data))
                    print(f"Adding to val: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile('nanoChatGPT/data/custom/train.bin')
val_ids.tofile('nanoChatGPT/data/custom/val.bin')
