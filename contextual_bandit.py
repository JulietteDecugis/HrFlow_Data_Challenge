import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from random import random, randint

class ContextualBandit:
    # 4 arms for: assistant, excutive, manager and director
    # context: employee + company embedding

    def __init__(self, num_arms, num_features):
        # alpha = num success, beta = num failures
        self.num_arms = num_arms
        self.num_features = num_features
        self.alpha = torch.ones((num_arms, num_features))
        self.beta = torch.ones((num_arms, num_features))

    def select_arm(self, contexts):
        sampled_means = torch.zeros(self.num_arms)
        for arm in range(self.num_arms):
            sampled_theta = torch.distributions.beta.Beta(self.alpha[arm, :], self.beta[arm, :]).sample()
            sampled_means[arm] = torch.sigmoid(torch.dot(sampled_theta, contexts))
        selected_arm = torch.argmax(sampled_means)
        return selected_arm.item()

    def update(self, arm_index, y):
        reward = self.get_reward(arm_index, y)
        self.alpha[arm_index, :] += reward * torch.ones(self.num_features)
        self.beta[arm_index, :] += (1 - reward) * torch.ones(self.num_features)
    
    def get_reward(self, arm_index, y):
        return arm_index == y.item()
        
def thompson_sampling(X,y, num_arms, num_epochs=10, epsilon=0):
    chosen_arms = []
    num_features = X.shape[1]
    bandit = ContextualBandit(num_arms, num_features)

    for i in tqdm(range(X.shape[0])):
        for i in range(num_epochs):
            context = X[i, :]
            p = random()
            if p < epsilon:
                chosen_arm = randint(0, num_arms-1)
            else:
                chosen_arm = bandit.select_arm(context)
            bandit.update(chosen_arm, y[i])
            chosen_arms.append(chosen_arm)
    return bandit, chosen_arms

def evaluate_bandit(X,y, bandit):
    acc = 0
    for i in range(X.shape[0]):
        arm = bandit.select_arm(X[i, :])
        acc += arm == y[i]
    return acc / X.shape[0]