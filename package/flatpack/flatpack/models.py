import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


def train_rnn_function(dataset, vocab_size, embed_size, hidden_size, num_layers, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.long()
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

        # Print epoch-wise progress
        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")

    return {'model': model}
