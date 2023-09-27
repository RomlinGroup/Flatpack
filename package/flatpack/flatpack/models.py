import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from .datasets import TextDataset


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(RNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size) if vocab_size is not None else None
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size) if vocab_size is not None else None

    @staticmethod
    def load_torch_model(model_path):
        return torch.load(model_path)

    def load_vocab_size(self, save_dir):
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)
        self.vocab_size = len(char_to_index)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        if self.embedding is None or self.fc is None:
            raise ValueError("vocab_size is not loaded")
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

    @classmethod
    def train_model(cls, indexed_text, seq_length, vocab_size, embed_size, hidden_size, num_layers, epochs, batch_size,
              device):
        """
        Train an RNN model with the given parameters and dataset.

        Args:
            indexed_text (list): The indexed text dataset.
            seq_length (int): The sequence length for training.
            vocab_size (int): The size of the vocabulary.
            embed_size (int): The embedding size.
            hidden_size (int): The hidden layer size.
            num_layers (int): The number of layers.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            device (torch.device): The device to train the model on.

        Returns:
            RNN: The trained RNN model.
        """
        dataset = TextDataset(indexed_text, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = cls(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
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
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

                _, predicted = torch.max(outputs.data, 2)
                correct = (predicted == targets)
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

    def generate_text(self, save_dir, start_sequence="To be, or not to be", generate_length=1024, temperature=1.0,
                      device=None):
        # Load char_to_index and index_to_char mappings from saved JSON files
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)

        with open(os.path.join(save_dir, 'index_to_char.json'), 'r') as f:
            index_to_char = json.load(f)

        input_sequence = [char_to_index[char] for char in start_sequence]
        input_tensor = torch.tensor(input_sequence).long().unsqueeze(0)

        if device is not None:
            input_tensor = input_tensor.to(device)

        generated_text = start_sequence

        self.eval()

        with torch.no_grad():
            for _ in range(generate_length):
                output = self(input_tensor)
                probabilities = F.softmax(output[0, -1] / temperature, dim=0)
                next_index = torch.multinomial(probabilities, 1).item()
                next_token = index_to_char[str(next_index)]  # JSON keys are always strings

                generated_text += next_token
                input_sequence = input_sequence[1:] + [next_index]
                input_tensor = torch.tensor(input_sequence).long().unsqueeze(0)

                if device is not None:
                    input_tensor = input_tensor.to(device)

        return generated_text
