import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader


class RNNLM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        # Defer initialization of the embedding layer until vocab_size is known
        self.embedding = None
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        # Defer initialization of the fc layer until vocab_size is known
        self.fc = None
        self.vocab_size = None

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

    def load_vocab(self, save_dir):
        with open(f'{save_dir}/char_to_index.json', 'r') as f:
            char_to_index = json.load(f)
        with open(f'{save_dir}/index_to_char.json', 'r') as f:
            index_to_char = json.load(f)
        self.vocab_size = len(char_to_index)
        self.embedding = nn.Embedding(self.vocab_size, self.rnn.input_size)
        self.fc = nn.Linear(self.rnn.hidden_size, self.vocab_size)
        return char_to_index, index_to_char

    def generate_text(self, save_dir, start_sequence="In the beginning", generate_length=1024, temperature=1.0):
        char_to_index, index_to_char = self.load_vocab(save_dir)
        input_sequence = [char_to_index[char] for char in start_sequence]
        input_tensor = torch.tensor(input_sequence).long().unsqueeze(0)
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

        return generated_text
