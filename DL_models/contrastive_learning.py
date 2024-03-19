import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from DL_models import LSTM, MLP, Transformer

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

class FusionModel(nn.Module):
    def __init__(self, input_size, num_layers, projected_size, hidden_size):
        super(FusionModel, self).__init__()
        self.text1 = Transformer(1, linear_size = hidden_size, d_model = input_size, 
                                 num_layers = num_layers,
                                 nhead = 4, classification_head=False)
        self.text2 = LSTM(projected_size, input_size, hidden_size, num_layers)
        self.w1 = nn.Parameter(torch.ones(projected_size, projected_size))
        self.w2 = nn.Parameter(torch.ones(projected_size, projected_size))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(projected_size, 2*projected_size)
        self.fc2 = nn.Linear(2*projected_size, projected_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb1 = self.text1(x[:, :32])
        emb2 = self.text2(x[:, 32:].unsqueeze(1))
        emb = emb1 @ self.w1 + emb2 @ self.w2
        # emb = emb / torch.norm(emb, dim = 1)
        emb = self.dropout(emb)
        emb = self.relu(emb)
        emb = self.relu(self.fc1(emb))
        return self.fc2(emb)
