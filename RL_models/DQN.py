import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from RL_env import Env
import gymnasium as gym
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
from preprocessing import get_data, dataframe_to_tensor, split_tensors
from torch.utils.data import DataLoader, random_split



class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        #print("Q", Q)
        return torch.argmax(Q).item()
    

class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model.double()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            Y = Y.double()
            A = A.double()
            X = X.double()
            #print(X.shape)
            #print(Y.shape)
            #print(A.shape)
            QYmax = self.model(Y).max(1)[0].detach()

            #print(QYmax.shape)
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            #print("X", X.shape)
            #print("A", A.shape)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset(True)
        epsilon = self.epsilon_max
        step = 0
        donebis=False
        greed_action_stay=0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
                greed_action_stay+=1
                if greed_action_stay==2:
                    donebis=True

            # step
            predicted_next_position, predicted_next_emb_state, reward, done, _ = env.step(action)
            #print("action", action)
            #print("next state", predicted_next_position)
            #print("reward", reward)
            #print("target",y_undersampled[env.index])
            self.memory.append(state, action, reward, predicted_next_emb_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done or donebis:
                greed_action_stay = 0
                donebis=False
                
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                if episode>=env.Emb.shape[0]:
                    state,_=env.reset(True)
                else:
                    state, _ = env.reset(False)
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = predicted_next_emb_state

        return episode_return
    
    import gymnasium as gym


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Declare network

state_dim = 65
n_action = 2
nb_neurons=128
DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)

# DQN config
config = {'nb_actions': 2,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 10000,
          'epsilon_min': 0.01,
          'epsilon_max': 1,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 64}

# Train agent
agent = dqn_agent(config, DQN.double())
path_to_X = "/Users/louisedurand-janin/Documents/GitHub/HrFlow_Data_Challenge/data/X_train.csv"
path_to_y = "/Users/louisedurand-janin/Documents/GitHub/HrFlow_Data_Challenge/data/y_train.csv"
X, y = get_data(path_to_X, path_to_y)
X, y = dataframe_to_tensor(X,y)
X_train, y_train, X_validation, y_validation = split_tensors(X,y)
print("Training set size ", X_train.shape[0])
print("Validation set size", X_validation.shape[0])
env = Env(X_train, y_train)
scores = agent.train(env, X_train.shape[0])
#plt.plot(scores)
# Evaluate on VAlIDATION
career_env_validation = Env(X_validation, y_validation)
pred=[]
y_pred = []
s,_ =  career_env_validation.reset(True)
i=0
for t in tqdm(range(len(y_validation))):
    for k in range(5):
        a = greedy_action(agent.model,s)

        predicted_next_position, predicted_next_emb_state, reward, d, _ = career_env_validation.step(a)

        s = predicted_next_emb_state
        if t in [0,1,2,3]:
            print("action chosen", a)

            print("s", predicted_next_position)
            print("target", y_validation[i])
        #if a ==0:
         #   break
        
        if d:
            break
    i+=1
    pred.append(s)
    y_pred.append(predicted_next_position.item())
    s,_=career_env_validation.reset(False)
print(f1_score(y_pred, np.array(y_validation), average='macro'))
with open('predictions_list.json', 'w') as file:
    json.dump(y_pred, file)