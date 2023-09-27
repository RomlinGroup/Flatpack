from .base import Base
import torch.nn as nn


class LSTM(Base):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(LSTM, self).__init__(embed_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        print("LSTM - Shape after embedding:", x.shape)
        out, _ = self.lstm(x)
        print("LSTM - Shape after LSTM:", out.shape)
        out = out.reshape(out.size(0) * out.size(1) * out.size(2), -1)
        print("LSTM - Shape after reshaping:", out.shape)
        out = self.fc(out)
        return out
