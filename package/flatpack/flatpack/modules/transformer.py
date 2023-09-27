from .base import Base
import torch.nn as nn


class Transformer(Base):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, vocab_size=None):
        super(Transformer, self).__init__(d_model, vocab_size)
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        print("Transformer - Shape of src after embedding:", src.shape)
        print("Transformer - Shape of tgt after embedding:", tgt.shape)
        out = self.transformer(src, tgt)
        print("Transformer - Shape after Transformer:", out.shape)
        out = out.reshape(out.size(0) * out.size(1), self.d_model)
        print("Transformer - Shape after reshaping:", out.shape)
        out = self.fc(out)
        return out
