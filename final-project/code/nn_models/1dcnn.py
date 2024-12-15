import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchviz import make_dot

def process_data(df):
    # Drop unused columns
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    # Rename column for convenience
    df = df.rename(columns={"csfractures": "csi"})
    target = df["csi"]
    df.drop(columns=["csi"], inplace=True)
    # Convert -1 in target to 0 (assuming binary classification)
    target.replace(-1, 0, inplace=True)
    # Encode controltype as dummy variables
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    # Keep only numeric columns
    df = df.select_dtypes(exclude=['object'])
    return df, target

class CNN_1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc_in = nn.Linear(num_features, 1024)
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc_hidden = nn.Linear(208, 16)
        self.fc_out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc_in(x)  
        x = x.view(x.size(0), 64, 16)
        x = x.transpose(1, 2) 
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc_hidden(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x

def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=32, device='cpu'):
    model.to(device)
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device)

    n_samples = X_train_tensor.size(0)
    num_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {epoch_loss / num_batches:.4f} "
              f"| Val Loss: {val_loss:.4f}")

def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    model.to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (outputs.squeeze() >= 0.5).long().cpu().numpy()
    
    y_true = y_test_tensor.cpu().numpy()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    recall = recall_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    
    return acc, f1, recall, cm

def plot_conv1d_filters(model, layer_name='conv'):
    """
    model: Your PyTorch model
    layer_name: Name of the Conv1d layer to visualize
    """
    # Get the layer by its name or attribute
    conv_layer = getattr(model, layer_name)  # e.g., model.conv1
    # Extract the weights
    weights = conv_layer.weight.detach().cpu().numpy()  
    # weights shape: (out_channels, in_channels, kernel_size)

    num_filters = weights.shape[0]
    fig, axes = plt.subplots(nrows=num_filters, figsize=(8, 2*num_filters))
    
    if num_filters == 1:
        axes = [axes]  # handle single filter case
    for i, ax in enumerate(axes):
        ax.plot(weights[i, 0, :])  # Plot the filter (assuming in_channels=1)
        ax.set_title(f'Filter {i}')
        ax.set_xlabel('Kernel Index')
        ax.set_ylabel('Weight Value')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sns.set_palette("colorblind")
    df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target, val_size=0.2, test_size=0.2, random_state=42
    )
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape to (batch_size, 1, num_features)
    num_features = X_train_resampled.shape[1]  # Ideally 300 for the param counts to match
    X_train_resampled = np.expand_dims(X_train_resampled, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Instantiate the new model
    model = CNN_1d(num_features=num_features)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model, optimizer, criterion,
        X_train_resampled, y_train_resampled,
        X_val, y_val,
        epochs=30, batch_size=32, device='cpu'
    )

    acc, f1, rec, cm = evaluate_model(model, X_test, y_test, device='cpu')
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    X_test_tensor = torch.from_numpy(X_test).float()
    output = model(X_test_tensor)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("cnn_architecture", format="pdf")
    

