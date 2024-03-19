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
from preprocessing import get_data, dataframe_to_tensor, split_tensors

# Function to collect samples offline
def collect_samples(env,  disable_tqdm=False, print_done_states=False):
    s, _ = env.reset(True)
    S = []
    A = []
    rewards = []
    S2 = []
    SA = []
    done= False
    D = []
    #print("s", s)
    #print("true position", env.y[env.index])

    for _ in tqdm(range(env.Emb.shape[0]-1), disable=disable_tqdm):
        #print("index", env.index)
        #print(len(S))
        while done==False:
            a = env.sample_action()
            s2, emb2s2, r, done, _ = env.step(a)
            #print("next state preidcted", s2)
            #print("next embedding state ", emb2s2)
            S.append(s)
            #print('shape S', len(S))
            A.append(a)
            sa = torch.cat([s, torch.tensor(a)])
            SA.append(sa)
            rewards.append(r)
            #print("reward", r)
            S2.append(emb2s2)
            D.append(done)
            #if done:
                #s, _ = env.reset(False)
                #print("new embedding", s)
                #print("true position", env.y[env.index])
                #print("done!")
            #else:
            s = emb2s2
        s, _ = env.reset(False)
        done = False
    S2 = np.array(S2)
    S = np.array(S)
    A = np.array(A)
    D = np.array(D)
    SA = np.array(SA)
    rewards = np.array(rewards)
    return S, A, rewards, S2, D, SA

def rf_fqi(S, A, R, S2, D, SA, iterations, nb_actions=2, gamma=0.9, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    #SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        #print("iteration", iter)
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                
                A2 = torch.full((S2.shape[0], 1), a2)
                #print(A2.shape)
                S2A2 = torch.cat([torch.from_numpy(S2), A2], dim=1)
                S2A2_array = np.array(S2A2)

                #print("shaoe A2S2", S2A2_array.shape)
                predictions = Qfunctions[-1].predict(S2A2_array)
                if iter==1:
                    print("action", a2)
                    print("predictions", predictions)
                    print("state2", S2)
                

                #print("shape pred", predictions.shape)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2_array)
            max_Q2 = np.max(Q2,axis=1)
            #print("max Q2",max_Q2[:10])
            print(np.dot((1-D.flatten()),max_Q2))
            value = R + gamma*(1-D.flatten())*max_Q2
            #print("cumulated reward", value)
            #print("value", value.shape)
        Q = RandomForestRegressor()
        Q.fit(SA,value)
        #print("cumulated reward", value)
        Qfunctions.append(Q)
    return Qfunctions


def greedy_action(Q,s,nb_actions=2):
    Qsa = []
    for a in range(nb_actions):
        #print("s", s)
        #print("a", torch.tensor([a]))
        sa = torch.cat([s, torch.tensor([a])], dim=0)
        sa = np.array(sa)
        #print(sa.reshape(-1,1).shape)
        #print("prediction", Q.predict(sa.reshape(1, -1)))
        Qsa.append(Q.predict(sa.reshape(1, -1)))
    #print("Qsa", Qsa)
    return np.argmax(Qsa)

path_to_X = "/Users/louisedurand-janin/Documents/GitHub/HrFlow_Data_Challenge/data/X_train.csv"
path_to_y = "/Users/louisedurand-janin/Documents/GitHub/HrFlow_Data_Challenge/data/y_train.csv"
X, y = get_data(path_to_X, path_to_y)
X, y = dataframe_to_tensor(X,y)
X_train, y_train, X_validation, y_validation = split_tensors(X,y)
print("Training set size ", X_train.shape[0])
print("Validation set size", X_validation.shape[0])
env = Env(X_train, y_train)
S, A, R, S2, D, SA = collect_samples(env)
qfunctions = rf_fqi(S, A, R, S2, D, SA, 2)
career_env_validation = Env(X_validation, y_validation)
pred=[]
y_pred = []
s,_ =  career_env_validation.reset(True)
i=0
#for t in tqdm(range(len(y_test))):
for t in tqdm(range(len(y_validation))):
    for k in range(6):
        a = greedy_action(qfunctions[-1],s)

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