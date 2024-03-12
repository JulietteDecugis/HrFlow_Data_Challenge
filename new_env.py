import torch
import gymnasium as gym 
import numpy as np

class DQNEnv(gym.Env):
    def __init__(self, Emb, EmbS, y):
        super(DQNEnv, self).__init__()
        self.Emb = Emb
        self.EmbS = EmbS
        self.y = y
        #self.num_agents = self.Emb.shape[0]
        self.state = torch.zeros(1)
        self.num_states = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
    
    def reward(self, predictions):
        return [1 if pred == true_value else 0 for pred, true_value in zip(predictions, self.y)]
    
    def action(self, position):
        if position == 3:
            return torch.Tensor([0])
        else:
            return torch.Tensor([0, 1])
    
    def sample_action(self):
        return torch.Tensor(torch.randint(0, self.state.shape[0], size = (1,)).item())
    
    def transition(self, state, action):
        state += action
        return state
    
    def step(self, action):
        # tuple(observation, reward, terminated, trunc = False, info = {},)
        observation = self.transition(self.state, action)
        reward = self.reward()
        terminated_n = observation_n == 3
        return (observation_n, reward_n, terminated_n, {})
    
    def reset(self):
        random_index = torch.randint(0, self.Emb.shape[0], (1,)).item()
        self.emb_selected = 
        self.state = torch.zeros(self.num_agents)
        return (self.state, {})
    def calculate_reward(true_next_position, predicted_next_position, current_position):
 
    
        # Récompenses et pénalités
        correct_prediction_reward = 10
        ambitious_prediction_reward = 5
        correct_stay_reward = 10
        wrong_prediction_penalty = -10

        if predicted_next_position == true_next_position:
            # Récompense pour une prédiction correcte de progression ou de non-progression
            return correct_prediction_reward
        elif predicted_next_position > current_position and predicted_next_position < true_next_position:
            # Récompense pour une prédiction ambitieuse mais plausible
            return ambitious_prediction_reward
        elif predicted_next_position == current_position and current_position == true_next_position:
            # Récompense pour une prédiction correcte de rester dans le poste actuel
            return correct_stay_reward
        else:
            # Pénalité pour une prédiction incorrecte
            return wrong_prediction_penalty
