from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size=None):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=embed_size * 4,
            max_position_embeddings=512
        )
        self.transformer = BertModel(config)
        self.fc = nn.Linear(embed_size, vocab_size) if vocab_size is not None else None

    @staticmethod
    def load_torch_model(model_path):
        return torch.load(model_path)

    def load_vocab_size(self, save_dir):
        with open(os.path.join(save_dir, 'char_to_index.json'), 'r') as f:
            char_to_index = json.load(f)
        self.vocab_size = len(char_to_index)
        self.fc = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x):
        if self.fc is None:
            raise ValueError("vocab_size is not loaded")

        # Create attention mask
        attention_mask = (x != 0).long()

        outputs = self.transformer(input_ids=x, attention_mask=attention_mask)
        out = self.fc(outputs.last_hidden_state)
        return out

    @classmethod
    def train_model(cls, indexed_text, vocab_size, seq_length, embed_size, num_heads, num_layers, epochs, batch_size,
                    device):
        dataset = TextDataset(indexed_text, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = cls(embed_size, num_heads, num_layers, vocab_size)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0
        for epoch in range(epochs):
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 2)
                correct = (predicted == targets).float()
                accuracy = correct.sum().item() / (targets.size(0) * targets.size(1))

                total_loss += loss.item()
                total_accuracy += accuracy
                total_batches += 1

            average_loss = total_loss / total_batches
            average_accuracy = total_accuracy / total_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")

        return {'model': model}

    def generate_text(self, save_dir, start_sequence="To be, or not to be", generate_length=1024, temperature=1.0,
                      device=None):
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
                next_token = index_to_char[str(next_index)]

                generated_text += next_token
                input_sequence = input_sequence[1:] + [next_index]
                input_tensor = torch.tensor(input_sequence).long().unsqueeze(0)

                if device is not None:
                    input_tensor = input_tensor.to(device)

        return generated_text
