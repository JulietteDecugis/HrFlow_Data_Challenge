import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

class ContextualBandit:
    # 4 arms for: assistant, excutive, manager and director
    # context: employee + company embedding
    # Beta prior 

    def __init__(self, num_arms, num_features):
        # alpha = num success, beta = num failures
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = torch.ones(num_arms, num_features)
        self.beta = torch.ones(num_arms, num_features)

    def select_arm(self, contexts):
        # take max from posterior with sampled theta from Beta dist
        theta = Beta(self.alpha, self.beta).sample()
        posterior = nn.Sigmoid(nn.ReLU(theta @ contexts))
        return torch.argmax(posterior).item()

    def update(self, arm_index, y):
        reward = self.get_reward(arm_index, y)
        self.alpha[arm_index,:] += reward
        self.beta[arm_index, :] += (1-reward)
    
    def get_reward(self, arm_index, y):
        return int(arm_index == y)