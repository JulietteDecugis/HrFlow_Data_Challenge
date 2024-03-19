# HrFlow_Data_Challenge

### Authors: Louise Durand--Janin & Juliette Decugis

Github to solve the data challenge presented by HrFlow.ai [https://challengedata.ens.fr/participants/challenges/151/]

Context: Can you predict the professional evolution of one employee ?

The data challenge presented by HRFlow.ai is a career prediction which relies on two BERT embedding: one representing employees and the other their company. 
They were generated from information about the employee, such as qualifications, experience, skills or interests, and details about the company, such as sector, type of business, management style or size.

The labels correspond to four positions in hierarchical order: "Assistant" < "Executive" < "Manager" < "Director".

For this challenge, it is assumed that an employee must progress linearly along hierarchical positions, meaning all Managers have previously been Assistants and Executives within the same company. Moreover, an employee cannot regress in term of position.

Our GitHub is organized as follows:
- DL_models: contrastive learning to create common embeddings, deep learning architectures and oridinal_classification
- data: original .csv files & their torch.tensor representations
- solutions: our submitted predictions for y_test
- RL_models: sequential learning approaches with FQI and DQN
- pre_processing.py: dataset creation, oversampling/undersampling solutions for balanced data
- loss_functions.py: CenterLoss (used for supervised constrastive learning) and FocalLoss (used for imbalanced data)






