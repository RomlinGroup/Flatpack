from flatpack.datasets import TextDataset
from torch.utils.data import DataLoader, Dataset

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_size, num_layers, max_length):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_length)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=ff_size)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


def train_model(dataset, vocab_size, embed_size, num_heads, ff_size, num_layers, epochs, batch_size, device):
    model = Transformer(vocab_size, embed_size, num_heads, ff_size, num_layers, max_length=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    return model


def generate_text(model, start_sequence, generate_length, temperature, device, char_to_index, index_to_char):
    model.eval()
    input_seq = [char_to_index[char] for char in start_sequence]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(1).to(device)

    generated_text = start_sequence
    with torch.no_grad():
        for _ in range(generate_length):
            output = model(input_seq)
            probabilities = torch.nn.functional.softmax(output[-1, 0] / temperature, dim=0)
            next_char_index = torch.multinomial(probabilities, 1).item()
            generated_text += index_to_char[next_char_index]

            input_seq = torch.cat([input_seq, torch.tensor([[next_char_index]], dtype=torch.long).to(device)])

    return generated_text


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, vocab_size, embed_size, num_heads, ff_size, num_layers):
    model = Transformer(vocab_size, embed_size, num_heads, ff_size, num_layers, max_length=64)
    model.load_state_dict(torch.load(path))
    return model
