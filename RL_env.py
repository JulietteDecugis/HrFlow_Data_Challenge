import torch
import gymnasium as gym 
import numpy as np

class Env(gym.Env):
    def __init__(self, Emb, y):
        super(Env, self).__init__()
        self.Emb = Emb
        self.y = y
        self.state = torch.zeros(1)
        self.index = 0 
        self.individual_emb = self.Emb[self.index]
        self.individual_target = self.y[self.index]
        self.individual_emb_state= torch.cat([self.individual_emb, self.state])
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
        action = self.action(self.state)
        return torch.Tensor(torch.randint(0, action.shape[0], size=(1,)))
    
    def transition(self, state, action):
        state += action
        return state
    
    def step(self, action):
        predicted_next_position = self.transition(self.state, action)
        predicted_next_emb_state = torch.cat([self.individual_emb, predicted_next_position ])
        reward = self.calculate_reward(predicted_next_position)
        terminated_n = predicted_next_position == 3
        return (predicted_next_position, predicted_next_emb_state, reward, terminated_n, {})
    
    def reset(self, first_emb):
        if first_emb:
            self.index =0
        else :
            self.index +=1
        self.state = torch.zeros(1)
        self.individual_emb = self.Emb[self.index]
        self.state = torch.zeros(1)
        self.individual_emb_state = torch.cat([self.individual_emb, self.state])
        return (self.individual_emb_state, {})
    
    def calculate_reward(self, predicted_next_position):
        true_next_position = self.y[self.index]
        current_position = self.state


    
        # Reward and penalties
        correct_prediction_reward = 1
        ambitious_prediction_reward = .5
        correct_stay_reward = 1
        wrong_prediction_penalty = -1

        if predicted_next_position == true_next_position:
            # Reward for correct prediction of progression or non-progression
            return correct_prediction_reward
        elif predicted_next_position > current_position and predicted_next_position < true_next_position:
            # Reward for an ambitious but plausible prediction
            return ambitious_prediction_reward
        elif predicted_next_position < true_next_position and predicted_next_position == current_position:
            return -0.5

        else:
            # Penalty for incorrect prediction
            return wrong_prediction_penalty
        