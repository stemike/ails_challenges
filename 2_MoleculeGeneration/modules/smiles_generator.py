import torch
from torch.autograd import Variable
from torch.nn import Module, LSTM, Linear, Embedding


class SmilesGenerator(Module):
    def __init__(self, one_hot_size, hidden_size=128, num_layers=1, padding_idx=0):
        super(SmilesGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = Embedding(one_hot_size, hidden_size, padding_idx=padding_idx)
        # seq, batch, feature
        self.model = LSTM(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=num_layers)
        self.decoder = Linear(hidden_size, one_hot_size)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return Variable(h), Variable(c)

    def forward(self, x, hidden):
        x = self.encoder(x)
        output, hidden = self.model.forward(x, hidden)
        output = self.decoder(output)
        return output, hidden
