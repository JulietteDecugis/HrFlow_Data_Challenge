import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class CLIP_Model(nn.Module):
    def __init__(self, emb1_size, emb2_size, h1_sizes = [32], h2_sizes = [32], type = 'MLP'):
        super(CLIP_Model, self).__init__()
        assert(h1_sizes[-1] == h2_sizes[-1])
        if type == 'MLP':
            self.text1 = MLP(emb1_size, h1_sizes)
            self.text2 = MLP(emb2_size, h2_sizes)
        elif type == 'LSTM':
            self.text1 = LSTM(h1_sizes[-1], 1, h1_sizes[0], len(h1_sizes), emb1_size)
            self.text2 = LSTM(h1_sizes[-1], 1, h1_sizes[0], len(h1_sizes), emb1_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def get_emb(self, emb1, emb2):
        return self.text1(emb1), self.text2(emb2)
    
    def forward(self, emb1, emb2, emb=True):
        # both MLPs - could change
        emb1 = self.text1(emb1)
        emb2 = self.text2(emb2)
        
        # normalize
        emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_emb = logit_scale * emb1 @ emb2.t()
        return logits_per_emb, emb1, emb2

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

class FusionModel(nn.Module):
    def __init__(self, input_size, num_layers, projected_size, hidden_size):
        super(FusionModel, self).__init__()
        self.text1 = LSTM(projected_size, input_size, hidden_size, num_layers)
        self.text2 = LSTM(projected_size, input_size, hidden_size, num_layers)
        self.w1 = nn.Parameter(torch.ones(projected_size, projected_size))
        self.w2 = nn.Parameter(torch.ones(projected_size, projected_size))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(projected_size, 2*projected_size)
        self.fc2 = nn.Linear(2*projected_size, projected_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb1 = self.text1(x[:, :, 0].unsqueeze(1))
        emb2 = self.text2(x[:, :, 1].unsqueeze(1))
        emb = emb1 @ self.w1 + emb2 @ self.w2
        # emb = emb / torch.norm(emb, dim = 1)
        emb = self.dropout(emb)
        emb = self.relu(emb)
        emb = self.relu(self.fc1(emb))
        return self.fc2(emb)

class Transformer(nn.Module):
    def __init__(self, linear_size, d_model, nhead=10):
        super().__init__()
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_feedforward = linear_size, d_model=d_model, nhead=nhead), num_layers=1),
        )

    def forward(self, x):
        return self.transformer(x.unsqueeze(0)).squeeze(0)