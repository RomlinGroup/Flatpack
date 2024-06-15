from torch.utils.data import Dataset

import requests
import torch


class TextDataset(Dataset):
    def __init__(self, indexed_text, seq_length):
        self.indexed_text = indexed_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.indexed_text) - self.seq_length

    def __getitem__(self, idx):
        inputs = torch.tensor(self.indexed_text[idx:idx + self.seq_length], dtype=torch.long)
        targets = torch.tensor(self.indexed_text[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return inputs, targets


def download_and_preprocess_text(url, limit=None):
    text = download_text(url)
    if limit is not None:
        text = text[:limit]
    chars = sorted(set(text))
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = dict(enumerate(chars))
    indexed_text = [char_to_index[char] for char in text]
    return indexed_text, char_to_index, index_to_char


def download_text(url):
    return requests.get(url).text
