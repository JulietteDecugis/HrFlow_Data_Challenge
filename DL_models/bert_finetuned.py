import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig
from tqdm import tqdm
from loss_functions import CenterLoss
from preprocessing import get_dataloaders

class Fine_Tuned_Bert(nn.Module):
    def __init__(self, input_size, num_classes):
        num_classes = 64
        super(Fine_Tuned_Bert, self).__init__()
        self.fc1 = nn.Linear(input_size, 768)
        self.fc = nn.Linear(768, num_classes)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', config=bert_config).encoder
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.model(x.unsqueeze(0)).last_hidden_state
        x = self.fc(x)
        # x = self.softmax(x)
        return x

def train(NUM_EPOCHS, dataloader, model, test_data):
    criterion = CenterLoss(4, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Training loop
    model.train()
    for epoch in tqdm(range(NUM_EPOCHS)):
        for inputs, labels in dataloader:
            # Forward pass
            predictions = model(inputs)[0]
            # Compute loss
            loss = criterion(labels, predictions)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            print('Epoch loss:', loss.item())
            # print('F1 train score: ', f1_score(torch.argmax(predictions, dim = 1), labels, average = 'macro'))
            X, y = next(iter(test_data))
            preds = model(X)[0]
            predTest = model(X)[0]
            print('Val loss ', criterion(y, X))
            # print('F1 test score: ', f1_score(torch.argmax(predTest, dim = 1), y, average = 'macro'))
            torch.save(model.state_dict(), 'bert_fine_tuned' + str(epoch) + '.pt')

def main(X, y, Xtest, ytest, embeddings=True):
    # if embeddings:
    #     fusion_model = FusionModel(input_size=32, num_layers=3, projected_size=32, hidden_size=64)
    #     fusion_model.load_state_dict(torch.load('fusion_model_128_3.pt'))
    #     fusion_model.eval()
    #     X = fusion_model(X)
    # focal_loss = FocalLoss(alpha = 0.9, gamma = 0.1)
    data = get_dataloaders(X, y, batch_size=128)
    test_data = get_dataloaders(Xtest, ytest, batch_size=128)
    model = Fine_Tuned_Bert(X.shape[1], 4)
    # Freeze the first 6 layers
    for param in model.parameters():
        param.requires_grad = False
    # Make sure the last few layers are trainable
    for i in range(4):
        for param in model.model.layer[-i].parameters():
            param.requires_grad = True
    train(10, data, model, test_data)
    torch.save(model.state_dict(), 'bert_fine_tuned.pt')

if __name__ == "__main__":
    X_train, y_train = torch.load('data/X_train_simple.pt'), torch.load('data/y_train.pt')
    cutoff = int(X_train.shape[0]*0.7)
    X_train1, y_train1 = X_train[:cutoff, :], y_train[:cutoff]
    X_val, y_val = X_train[cutoff:, :], y_train[cutoff:]
    X_train, y_train = X_train1, y_train1
    X_train.shape, X_val.shape
    # X_test, y_test = torch.load('data/X_test_simple.pt'), torch.load('data/y_test.pt')
    main(X_train, y_train, X_val, y_val)