import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from DL_models.architectures import Transformer, LSTM
from sklearn.metrics import f1_score
from loss_functions import FocalLoss
from preprocessing import get_dataloaders
        
class Ordinal_Classifier:
    """
    Creates classifers for P(class > 0), P(class > 1), P(class > 2)...
    Generates probabilities by combining these models.

    Args:
        num_classifiers (int): number of probability classifiers, should be equal to number of ordinal classes - 1
        num_features (int): number of input features for each data point
        criterion (list): list of loss functions for each classifier (Pytorch nn.Module)
        epochs: number of epochs to train each classifier at initialization
        params (list): list of parameter dict for each classifier 
    """

    def __init__(self, num_classifiers, num_features, criterions, epochs, params):
        self.epochs = epochs
        self.num_classifiers = num_classifiers
        self.num_features = num_features
        self.classifiers = [Proba_Classifier(params[i], criterions[i]) for i in range(self.num_classifiers)]
    
    def initialize(self, dataloaders):
        '''
        Args:
            dataloaders: list of torch.Dataloaders with training X and y
        '''
        for i in range(len(self.num_classifiers)):
            classifier = self.classifiers[i]
            dataloader = dataloaders[i]
            classifier.train(dataloader, num_epochs=self.epochs[i], num=i)      
        
    def get_scores(self, x):
        # x: torch.tensor with features, has size (1, num_features)
        # assumes our loss function returns positive score, negative score (order matters)
        # All classifiers are binary
        # We have p_gi_1: P(X > i) = 1 and p_gi_0 = P(X > i) = 0

        assert(self.num_classifiers > 0)
        scores = []
        p_g0_1, p_class0 = torch.sigmoid(self.classifiers[0].model(x))
        prev_class_positive = p_g0_1
        scores.append(p_class0.item())
        for i in range(1, len(self.num_classifiers-1)):
            p_gi_1, current_negative = torch.sigmoid(self.classifiers[i].model(x))
            p_class_i = prev_class_positive * current_negative
            prev_class_positive = p_gi_1
            scores.append(p_class_i.item())
        
        # for exmaple: P(class = 4) = P(class > 3)
        if self.num_classifiers > 1:
            scores.append(prev_class_positive)
        
        # return most probable class:
        return np.argamx(scores)
    
class Proba_Classifier():
    def __init__(self, params, criterion):
        super(Proba_Classifier).__init__()
        self.type = params['type']
        if self.type == 'Transformer':
            self.model = Transformer(params['num_classes'], params['hidden_size'], params['input_size'], params['num_layers'], params['n_head'])
        elif self.type == 'LSTM':
            self.model = LSTM(params['num_classes'], params['input_size'], params['hidden_size'], params['num_layers'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params['lr'])
        self.criterion = criterion
    
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
                loss = self.criterion(probs, y)
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

def get_params(**params):
    default_params = {'num_classes': 2,
              'hidden_size': 12,
              'input_size': 32,
              'num_layers': 2,
              'n_head': 4,
              'type': 'Transformer'}
    for item in params.keys():
        default_params[item] = params[item]
    loss_g0 = FocalLoss(alpha = 0.3, gamma = 0)
    loss_g1 = FocalLoss(alpha = 0.7, gamma = 0)
    loss_g2 = FocalLoss(alpha = -1, gamma=0)
    criterions = [loss_g0, loss_g1, loss_g2]
    return params, criterions

def processing(X, y):
    encoding_map = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
    }
    dataloaders = []
    data_X = X

    # generate binary labels
    for i in range(3):
        encoding_map[i] = 0
        yi = [encoding_map[category.item()] for category in y]
        yi = torch.tensor(yi)
        dataloaders.append(get_dataloaders(X, yi, batch_size=128))
    return dataloaders

if __name__ == "__main__":
    # create datasets
    X_train, y_train = torch.load('data/X_train_simple.pt'), torch.load('data/y_train.pt')
    cutoff = int(X_train.shape[0]*0.7)
    X_train1, y_train1 = X_train[:cutoff, :], y_train[:cutoff]
    X_val, y_val = X_train[cutoff:, :], y_train[cutoff:]
    X_train, y_train = X_train1, y_train1
    dataloaders = processing(X_train, y_train)

    # create loss functions & architecture parameters
    params, criterions = get_params()
    num_models = 3
    hidden_size = 64
    ensemble_model = Ordinal_Classifier(num_models, hidden_size, params=params, epochs=[20, 20, 20], criterions=criterions)
    
    # train all models
    ensemble_model.initialize(dataloaders)

    # evaluate on the validation set:
    y_val_pred = []
    for i in range(X_val.shape[0]):
        val = X_val[i, :]
        pred = ensemble_model.select_arm(val)
        y_val_pred.append(pred)
    print('Validation F1 score:', f1_score(y_val, y_val_pred, average = 'macro'))