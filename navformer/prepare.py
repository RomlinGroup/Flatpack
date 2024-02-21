# Based on https://github.com/VatsaDev/nanoChatGPT/blob/main/data/Chat/prepare.py by VatsaDev (MIT license)

import os
import re
import numpy as np
import tiktoken


def split_file(filename, output_dir, chunk_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), chunk_size):
        output_filename = os.path.join(output_dir, f'{i // chunk_size}-dataset.txt')
        with open(output_filename, 'w') as f:
            f.writelines(lines[i:i + chunk_size])


def get_file_number(filename):
    match = re.match(r"(\d+)-dataset\.txt", filename)
    return int(match.group(1)) if match else None


def process_files(input_dir, train_proportion=0.8):
    output_files = sorted(filter(lambda f: f.endswith('.txt'), os.listdir(input_dir)))
    n_files = len(output_files)
    train_file_limit = int(n_files * train_proportion)

    print(f"Total files: {n_files}, Training file limit: {train_file_limit}")

    train_ids, val_ids = [], []
    enc = tiktoken.get_encoding("gpt2")

    for filename in output_files:
        file_number = get_file_number(filename)
        if file_number is not None:
            try:
                with open(os.path.join(input_dir, filename), 'r') as f:
                    data = f.read()
                target_list = train_ids if file_number < train_file_limit else val_ids
                target_list.extend(enc.encode_ordinary(data))
                print(f"Adding to {'train' if target_list is train_ids else 'val'}: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    return np.array(train_ids, dtype=np.uint16), np.array(val_ids, dtype=np.uint16)


split_file('dataset.txt', 'nanoChatGPT/output', 1000)
train_ids, val_ids = process_files('nanoChatGPT/output')

train_ids.tofile('nanoChatGPT/data/custom/train.bin')
val_ids.tofile('nanoChatGPT/data/custom/val.bin')
