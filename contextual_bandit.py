import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta
from Fusion_Model import Transformer, LSTM
from sklearn.metrics import f1_score

class ContextualBandit:
    # 4 arms for: assistant, excutive, manager and director
    # context: employee + company embedding
    # Beta prior 

    def __init__(self, num_arms, num_features, criterions, epochs, params):
        self.epochs = epochs
        # alpha = num success, beta = num failures
        self.num_arms = num_arms
        self.num_features = num_features
        params_LSTM = {
            'num_classes': 2,
            'hidden_size': 64,
            'input_size': self.num_features,
            'num_layers': 2,
            'type': 'LSTM'
        }
        params_Transformer = {
            'num_classes': 2,
            'hidden_size': 128,
            'input_size': self.num_features,
            'num_layers': 3,
            'n_head': 4,
            'type': 'Transformer'
        }
        self.arms = [Arm_Model(params[0], 1e-3, criterions[0]), 
                     Arm_Model(params[1], 1e-5, criterions[1]),
                     Arm_Model(params[2], 1e-5, criterions[2])]
    
    def initialize(self, dataloaders):
        for i in range(len(self.arms)):
            print(i)
            arm = self.arms[i]
            dataloader = dataloaders[i]
            arm.train(dataloader, num_epochs=self.epochs[i], num=i)         
        
    def select_arm(self, X):
        # take max from posterior with sampled theta from Beta dist
        # assumes batch size == 1 
        # class 0 if p(x > 1) = 0
        p_g0_1, p_class0 = torch.sigmoid(self.arms[0].model(X))
        p_g1_1, p_g1_0 = torch.sigmoid(self.arms[1].model(X))
        p_class1 = p_g0_1 * p_g1_0
        p_g2_1, p_g2_0 = torch.sigmoid(self.arms[2].model(X))
        p_class2 = p_g1_1 * p_g2_0
        p_class3 = p_g2_1
        all_scores = [p_class0, p_class1, p_class2, p_class3]
        all_scores = [a.item() for a in all_scores]
        # p_class2 = p_class2.item()
        # p_class3 = p_g2_1.item()
        return np.argmax(all_scores)

        # if p_greater_1 == 0:
        #     return 0
        # else:
        #     p_greater_1 = self.arms[1](X)
        #     p_less_than_3 = self.arms[2](X)
        #     class1 = p_            
        # values = []
        # labels = []

        # for i in range(self.num_arms):
        #     curr_arm = self.arms[i]
        #     max_prob, label = curr_arm.prediction(X)
        #     max_prob, label = max_prob.item(), label.item()
        #     values.append(max_prob)
        #     labels.append(label)
        # if 1 in labels:
        #     values, labels = pd.Series(values), pd.Series(labels)
        #     return values.loc[labels[labels == 1].index].idxmax()
        # else:
        #     return np.argmin(values)

    # def update(self, arm_index, y):
    #     _, label = self.arms[arm_index].prediction(X)
    #     reward = self.get_reward(arm_index, y)
    #     self.alpha[arm_index,:] += reward
    #     self.beta[arm_index, :] += (1-reward)
    
    # def get_reward(self, arm_index, y):
    #     return arm_index

class Arm_Model():
    def __init__(self, params, lr, criterion):
        super().__init__()
        self.type = params['type']
        if self.type == 'Transformer':
            self.model = Transformer(params['num_classes'], params['hidden_size'], params['input_size'], params['num_layers'], params['n_head'])
        elif self.type == 'LSTM':
            self.model = LSTM(params['num_classes'], params['input_size'], params['hidden_size'], params['num_layers'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.criterions = criterion
    
    def train(self, dataloader, num_epochs, num):
        self.model.train()
        losses = []
        for _ in tqdm(range(num_epochs)):
            loss_epoch = []
            for batch in dataloader:
                X, y = batch
                X = X.float()
                if self.type == 'LSTM':
                    X = X.reshape(X.shape[0], 1, X.shape[1])
                self.optimizer.zero_grad()
                probs = self.model(X)
                loss = self.criterions(probs, y)
                loss.backward()
                self.optimizer.step()
                loss_epoch.append(loss.item())
            losses.append(np.mean(loss_epoch))
        print('Final loss: ', losses[-1])
        print('F1 score: ', f1_score(y, self.prediction(X)[1], average = 'macro'))
        np.save('losses' + str(num) + '.npy', losses)
        self.model.eval()
    
    def prediction(self, X):
        self.model.eval()
        if self.type == 'LSTM' and len(X.shape) < 3:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        with torch.no_grad():
            probs = self.model(X)
            max_object = torch.max(probs, dim=1)
            max_prob, y_pred = max_object[0], max_object[1]
        return max_prob, y_pred