import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from Lstm import Lstm
import Config as config


class Embedder(nn.Module):
    # 加一个is_decoder，便于在encode
    def __init__(self, vocab_size, d_model, is_encoder=None):
        super().__init__()
        self.d_model = d_model
        self.is_encoder = is_encoder
        self.embed = nn.Embedding(vocab_size, d_model)
        if (config.global_overview and self.is_encoder):
            self.lstm = Lstm(config.d_model, config.d_model, config.lstm_layers)

    def forward(self, x):
        x = self.embed(x)
        if (config.global_overview and self.is_encoder):
            lstm_out, _ = self.lstm(x)
            x = torch.cat((lstm_out[:, -1, :].unsqueeze(1), x), dim=1)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=2000, dropout=0.1):
    # def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
