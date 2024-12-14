import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils import resample

sns.set_palette("colorblind")
df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)
target = pd.read_csv('../data/target.csv')
target.columns = target.columns.str.lower()

target.rename(columns={"csfractures": "csi"}, inplace=True)
target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()

class MLP(torch.nn.Module):
    def __init__(self, input_size, units, dropout_list, output_size):
        super(MLP, self).__init__()
        self.units = units
        for i, u in enumerate(units):
            setattr(self, f'fc{i}', torch.nn.Linear(input_size, u))
            setattr(self, f'dropout{i}', torch.nn.Dropout(p=dropout_list[i]))
            setattr(self, f'relu{i}', torch.nn.ReLU())
            # setattr(self, f'batchnorm{i}', torch.nn.BatchNorm1d(u))
            input_size = u
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_out = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        for i in range(len(self.units)):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'relu{i}')(x)
            # x = getattr(self, f'batchnorm{i}')(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x

def process_data(df, target):
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    df = df.select_dtypes(exclude=['object'])
    return df, target

if __name__ == "__main__":
    df, target = process_data(df, target)
    # X_train, X_val, X_test, y_train, y_val, y_test = site_train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # Combine X and y for resampling
    train_data = pd.concat([pd.DataFrame(X_train.numpy()), pd.DataFrame(y_train.numpy(), columns=['target'])], axis=1)
    
    # Separate majority and minority classes
    majority_class = train_data[train_data.target == 0]
    minority_class = train_data[train_data.target == 1]
    
    # Upsample minority class
    minority_upsampled = resample(minority_class, 
                                  replace=True,     # sample with replacement
                                  n_samples=len(majority_class),    # to match majority class
                                  random_state=42) # reproducible results
    
    # Combine majority class with upsampled minority class
    upsampled_train_data = pd.concat([majority_class, minority_upsampled])
    
    # Separate X and y
    X_train = torch.tensor(upsampled_train_data.drop('target', axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(upsampled_train_data['target'].values, dtype=torch.float32)
    
    input_size = X_train.shape[1]
    output_size = 1
    units = [64, 32, 16]
    dropout_list = [0.5, 0.3, 0.1]
    model = MLP(input_size, units, dropout_list, output_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 100
    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            y_val_pred = (y_val_pred > 0.5).float()
            y_val_pred = y_val_pred.numpy().ravel()
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            recall = recall_score(y_val, y_val_pred)
            print(f"Epoch: {epoch}, val_acc: {val_acc}, val_f1: {val_f1}, recall: {recall}")