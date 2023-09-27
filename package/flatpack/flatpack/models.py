import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Constants
LEARNING_RATE = 0.001
START_SEQUENCE = "To be, or not to be"
GENERATE_LENGTH = 1024
TEMPERATURE = 1.0
MAX_NORM = 1


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(LSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.fc = nn.Linear(hidden_size, vocab_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    @classmethod
    def load_torch_model(cls, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = cls(checkpoint['embed_size'], checkpoint['hidden_size'], checkpoint['num_layers'],
                    checkpoint['vocab_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_torch_model(self, model_path):
        torch.save({
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'state_dict': self.state_dict(),
        }, model_path)

    def load_vocab_size(self, save_dir):
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)
        self.vocab_size = len(char_to_index)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        if self.embedding is None:
            raise ValueError("Embedding is not initialized")
        if self.fc is None:
            raise ValueError("Fully connected layer is not initialized")
        x = self.embedding(x)
        out, _ = self.lstm(x)  # LSTM returns a tuple (output, (hn, cn))
        out = self.fc(out)
        return out

    def train_model(self, dataset, epochs, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        total_correct = 0
        total_predictions = 0

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for inputs, targets in dataloader:
                inputs = inputs.long().to(device)
                targets = targets.long().to(device)
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))

                _, predicted = torch.max(outputs.data, 2)
                correct = (predicted == targets)
                total_correct += correct.sum().item()
                total_predictions += targets.numel()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=MAX_NORM)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            average_loss = total_loss / total_batches
            accuracy = total_correct / total_predictions
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {
            'model': self,
            'char_to_index': dataset.char_to_index,
            'index_to_char': dataset.index_to_char
        }

    def generate_text(self, save_dir, start_sequence=START_SEQUENCE, generate_length=GENERATE_LENGTH,
                      temperature=TEMPERATURE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)

        with open(os.path.join(save_dir, 'index_to_char.json'), 'r') as f:
            index_to_char = {int(k): v for k, v in json.load(f).items()}

        input_sequence = [char_to_index[char] for char in start_sequence]
        input_tensor = torch.tensor(input_sequence).long().unsqueeze(0).to(device)
        generated_text = start_sequence

        self.eval()

        with torch.no_grad():
            for _ in range(generate_length):
                output = self(input_tensor)
                probabilities = F.softmax(output[0, -1] / temperature, dim=0)
                next_index = torch.multinomial(probabilities, 1).item()
                next_token = index_to_char[next_index]

                generated_text += next_token
                input_sequence = input_sequence[1:] + [next_index]
                input_tensor = torch.tensor(input_sequence).long().unsqueeze(0).to(device)

        return generated_text


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(RNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.fc = nn.Linear(hidden_size, vocab_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

    @classmethod
    def load_torch_model(cls, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = cls(checkpoint['embed_size'], checkpoint['hidden_size'], checkpoint['num_layers'],
                    checkpoint['vocab_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_torch_model(self, model_path):
        torch.save({
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'state_dict': self.state_dict(),
        }, model_path)

    def load_vocab_size(self, save_dir):
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)
        self.vocab_size = len(char_to_index)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        if self.embedding is None:
            raise ValueError("Embedding is not initialized")
        if self.fc is None:
            raise ValueError("Fully connected layer is not initialized")
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

    def train_model(self, dataset, epochs, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        total_correct = 0
        total_predictions = 0

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for inputs, targets in dataloader:
                inputs = inputs.long().to(device)
                targets = targets.long().to(device)
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))

                _, predicted = torch.max(outputs.data, 2)
                correct = (predicted == targets)
                total_correct += correct.sum().item()
                total_predictions += targets.numel()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=MAX_NORM)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            average_loss = total_loss / total_batches
            accuracy = total_correct / total_predictions
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {
            'model': self,
            'char_to_index': dataset.char_to_index,
            'index_to_char': dataset.index_to_char
        }

    def generate_text(self, save_dir, start_sequence=START_SEQUENCE, generate_length=GENERATE_LENGTH,
                      temperature=TEMPERATURE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)

        with open(os.path.join(save_dir, 'index_to_char.json'), 'r') as f:
            index_to_char = {int(k): v for k, v in json.load(f).items()}

        input_sequence = [char_to_index[char] for char in start_sequence]
        input_tensor = torch.tensor(input_sequence).long().unsqueeze(0).to(device)
        generated_text = start_sequence

        self.eval()

        with torch.no_grad():
            for _ in range(generate_length):
                output = self(input_tensor)
                probabilities = F.softmax(output[0, -1] / temperature, dim=0)
                next_index = torch.multinomial(probabilities, 1).item()
                next_token = index_to_char[next_index]

                generated_text += next_token
                input_sequence = input_sequence[1:] + [next_index]
                input_tensor = torch.tensor(input_sequence).long().unsqueeze(0).to(device)

        return generated_text
