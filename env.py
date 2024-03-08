import torch
import gymnasium as gym 
import numpy as np

class MultiAgentEnv(gym.Env):
    def __init__(self, X, y):
        super(MultiAgentEnv, self).__init__()
        self.X = X
        self.y = y
        self.num_agents = self.X.shape[0]
        self.state = torch.zeros(self.num_agents)
        self.num_states = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
    
    def reward(self, pred):
        return pred == self.y
    
    def action(self, position):
        if position == 3:
            return torch.Tensor([0])
        else:
            return torch.Tensor([0, 1])
    
    def sample_action(self):
        return torch.Tensor([torch.randint(0, self.action(p).shape[0], size = (1,)).item() for p in self.state])
    
    def transition(self, state, action):
        state += action
        return state
    
    def step(self, action_n):
        # tuple(observation, reward, terminated, trunc = False, info = {},)
        observation_n = self.transition(self.state, action_n)
        reward_n = self.reward(observation_n)
        terminated_n = observation_n == 3
        return (observation_n, reward_n, terminated_n, {})
    
    def reset(self):
        self.state = torch.zeros(self.num_agents)
        return (self.state, {})