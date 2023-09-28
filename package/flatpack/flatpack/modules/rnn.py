from .base import Base
import torch.nn as nn


class RNN(Base):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size=None):
        super(RNN, self).__init__(embed_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

        # Override the initialization of the self.fc layer with the correct input feature size
        if vocab_size is not None:
            self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        print("RNN - Shape after embedding:", x.shape)
        out, _ = self.rnn(x)
        print("RNN - Shape after RNN:", out.shape)
        out = out.reshape(out.size(0) * out.size(1), -1)
        print("RNN - Shape after reshaping:", out.shape)
        out = self.fc(out)
        return out
