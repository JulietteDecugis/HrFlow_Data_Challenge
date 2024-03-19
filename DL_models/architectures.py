import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.layers = []
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.layers:
            x = self.relu(layer(x))
        return x

class LSTM(nn.Module):
    # for us: 2 seq with 64 length
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class Transformer(nn.Module):
    def __init__(self, num_classes, linear_size, d_model, num_layers = 1, nhead=10, classification_head=True):
        super().__init__()
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_feedforward = linear_size, d_model=d_model, nhead=nhead), num_layers=num_layers),
        )
        self.classification_head = classification_head
        if classification_head:
            self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        if not self.classification_head:
            return x
        else:
            x = self.fc(x)
            return x
