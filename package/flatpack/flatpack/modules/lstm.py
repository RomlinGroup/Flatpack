from base import Base
import torch.nn as nn


class LSTM(Base):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(LSTM, self).__init__(embed_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
