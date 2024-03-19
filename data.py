import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

def get_data(path_to_X, path_to_y):
    '''
    Function that takes the paths to X and y 
    Deserializes emebddings in X and encodes positions in y 
    '''
    y = pd.read_csv(path_to_y, index_col=0)
    encoding_map = {
        "Assistant": 0,
        "Executive": 1,
        "Manager": 2,
        "Director": 3,
    }

    y = np.array([encoding_map[category] for category in y['position']])

    X = pd.read_csv(path_to_X, index_col=0)
    X['employee embedding'] = X['employee embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    X['company embedding'] = X['company embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))

    return X, y

def dataframe_to_tensor(X, y, encoded=True):
    '''
    Function that takes 2 dataframe X and target y  and a boolean encoded that indicates if y is encoded
    as arguments and returns the associated tensors 
    '''
    # Convert embeddings columns in PyTorch tensors 
    employee_embedding_tensor = torch.tensor(np.vstack(X['employee embedding'].values), dtype=torch.float64)
    company_embedding_tensor = torch.tensor(np.vstack(X['company embedding'].values), dtype=torch.float64)

    # Concatenate both
    combined_tensor = torch.cat([employee_embedding_tensor, company_embedding_tensor], dim=1)
    if not encoded:
        encoding_map = {
            "Assistant": 0,
            "Executive": 1,
            "Manager": 2,
            "Director": 3,
        }

        y = torch.tensor(np.array([encoding_map[category] for category in y['position']]))
        return combined_tensor, y
    y = torch.tensor(y)
    return combined_tensor, y

def embeddings_reduced_PCA(X, reduced_length):
    '''
    Function that reduces company embeddings to reduced_length thanks to PCA

    output : tensor resulting from the concatenation of employee emebddings and reduced company embeddings
    '''
    #Reduce company embeddings to length reduced_length
    embeddings_company = np.vstack(X['company embedding'].values)


    # Create PCA instance
    pca = PCA(n_components=reduced_length)

    # Fit et transformation des embeddings
    reduced_embeddings = pca.fit_transform(embeddings_company)

    # Convert embeddings columns in PyTorch tensors
    employee_embedding_tensor = torch.tensor(np.vstack(X['employee embedding'].values), dtype=torch.float64)
    company_embedding_tensor = torch.tensor(reduced_embeddings, dtype=torch.float64)

    # Concatenate both
    combined_tensor = torch.cat([employee_embedding_tensor, company_embedding_tensor], dim=1)
    return combined_tensor
    

def data_augmentation_emixup(path_to_Xtrain, path_to_ytrain, nb_augmentation):
    '''
    Function that makes data augmentation on the embeddings based on the EMixup method from https://arxiv.org/pdf/1912.00772.pdf
    '''
    X_train = pd.read_csv(path_to_Xtrain, index_col=0)
    X_train['employee embedding'] = X_train['employee embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    X_train['company embedding'] = X_train['company embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    X_train.drop("id", axis=1, inplace=True)
    y_train = pd.read_csv(path_to_ytrain, index_col=0)
    encoding_map = {
        "Assistant": 0,
        "Executive": 1,
        "Manager": 2,
        "Director": 3,
    }

    y_train = np.array([encoding_map[category] for category in y_train['position']])
    mask_3 = (y_train == 3)
    indices_3 = X_train.iloc[mask_3]
    mask_2 = (y_train == 2)
    indices_2 = X_train.iloc[mask_2]
    # Parameter of the Beta law
    alpha = 2.0  
    beta = 5.0   
    for i in range(nb_augmentation):
        random_index1 = torch.randint(0, indices_3.shape[0], (1,)).item()  # Get an indice randomly
        random_line1 = indices_3.iloc[random_index1]
        random_index2 = torch.randint(0, indices_3.shape[0], (1,)).item()
        random_line2 = indices_3.iloc[random_index2]


        # Generate one sample of the beta distribution
        sample = np.random.beta(alpha, beta)
        new_line_company = sample*random_line1["company embedding"] + (1-sample)*random_line2["company embedding"]
        new_line_employee = sample*random_line1["employee embedding"] + (1-sample)*random_line2["employee embedding"]
        new_index = X_train.index.max() + 1

        # Add the new line 
        new_line_series = pd.Series([new_line_employee, new_line_company], index=X_train.columns, name=new_index)

        X_train = pd.concat([X_train, pd.DataFrame(new_line_series).T])
        y_train = np.append(y_train, 3)

        # Same for target 2 : Manager
        random_index1 = torch.randint(0, indices_2.shape[0], (1,)).item()  
        random_line1 = indices_2.iloc[random_index1]
        random_index2 = torch.randint(0, indices_2.shape[0], (1,)).item()
        random_line2 = indices_2.iloc[random_index2]



        sample = np.random.beta(alpha, beta)
        new_line_company = sample*random_line1["company embedding"] + (1-sample)*random_line2["company embedding"]
        new_line_employee = sample*random_line1["employee embedding"] + (1-sample)*random_line2["employee embedding"]
        new_index = X_train.index.max() + 1
        new_line_series = pd.Series([new_line_employee, new_line_company], index=X_train.columns, name=new_index)


        X_train = pd.concat([X_train, pd.DataFrame(new_line_series).T])
        y_train = np.append(y_train, 2)

    return X_train, y_train

def undersampling(tensor_x, tensor_y):
    '''
    input : tensor_x : Tensor of combined embeddings
            tensor_y : Tensor of encoded target
    
    output : Tensor dataset undersampled and corresponding target tensor 
    '''
    undersample = NearMiss(version=1, n_neighbors=3)
    X_undersampled, y_undersampled = undersample.fit_resample(np.array(tensor_x), np.array(tensor_y))
    # Shuffle 
    X_undersampled, y_undersampled = shuffle(X_undersampled, y_undersampled, random_state=42)

    return torch.tensor(X_undersampled), torch.tensor(y_undersampled)

def oversampling(tensor_x, tensor_y):
    '''
    input : tensor_x : Tensor of combined embeddings
            tensor_y : Tensor of encoded target
    
    output : Tensor dataset oversampled and corresponding target tensor 
    '''
    ros = RandomOverSampler(random_state=0)
    X_oversampled, y_oversampled = ros.fit_resample(tensor_x, tensor_y)
    X_oversampled, y_oversampled = shuffle(X_oversampled, y_oversampled, random_state=42)
    return torch.tensor(X_oversampled), torch.tensor(y_oversampled)
