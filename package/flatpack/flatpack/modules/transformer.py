from .base import Base
import torch.nn as nn


class Transformer(Base):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, vocab_size=None, **kwargs):
        super(Transformer, self).__init__(d_model, vocab_size)
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt)
        out = out.reshape(out.size(0) * out.size(1), -1)
        out = self.fc(out)
        return out
