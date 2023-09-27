import requests
import torch
from torch.utils.data import Dataset


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


def prepare_text_dataset(url, seq_length=64, subset_size=None):
    text = requests.get(url).text

    if subset_size is not None:
        text = text[:subset_size]

    chars = sorted(set(text))
    indexed_text = [chars.index(char) for char in text]
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for i, char in enumerate(chars)}
    dataset = TextDataset(indexed_text, seq_length=seq_length)
    return dataset, char_to_index, index_to_char
