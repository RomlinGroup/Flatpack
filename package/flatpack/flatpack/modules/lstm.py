from .base import Base
import torch.nn as nn


class LSTM(Base):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None, **kwargs):
        super(LSTM, self).__init__(embed_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        if vocab_size is not None:
            self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0) * out.size(1), -1)
        out = self.fc(out)
        return out
