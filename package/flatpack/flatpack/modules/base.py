import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from flatpack.datasets import TextDataset


class Base(nn.Module):
    def __init__(self, embed_size, vocab_size=None):
        super(Base, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size) if vocab_size is not None else None
        self.fc = nn.Linear(self.embed_size, self.vocab_size) if vocab_size is not None else None

    @staticmethod
    def load_torch_model(model_path):
        return torch.load(model_path)

    def load_vocab_size(self, save_dir):
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)
        self.vocab_size = len(char_to_index)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.fc = nn.Linear(self.embed_size, self.vocab_size)

    @classmethod
    def train_model(cls, indexed_text, seq_length, vocab_size, epochs=100, batch_size=64, device='cpu', **kwargs):
        from .transformer import Transformer
        dataset = TextDataset(indexed_text, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = cls(vocab_size=vocab_size, **kwargs)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            total_batches = 0

            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                if isinstance(model, Transformer):
                    outputs = model(targets, targets)
                else:
                    outputs = model(inputs)

                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == targets.view(-1))
                accuracy = correct.sum().item() / (targets.size(0) * targets.size(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy
                total_batches += 1

            average_loss = total_loss / total_batches
            average_accuracy = total_accuracy / total_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")

        return {'model': model}
