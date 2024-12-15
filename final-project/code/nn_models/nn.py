import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_val_test_split, process_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

class MLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(1)  # shape: (batch_size, num_features)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
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
    precison = precision_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    
    return acc, f1, recall, cm, precison

def plot_conv1d_filters(model, layer_name='conv'):
    """
    model: Your PyTorch model
    layer_name: Name of the Conv1d layer to visualize
    """
    # This function was specific to Conv1d layers.
    # We'll keep it here for consistency but note that
    # it does not apply to the MLP model.
    conv_layer = getattr(model, layer_name, None)
    if conv_layer is None:
        print("No such layer found in model. Skipping visualization.")
        return
    weights = conv_layer.weight.detach().cpu().numpy()  
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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv('../../data/merged_data_cleaned.csv', low_memory=False)

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
    num_features = X_train_resampled.shape[1]
    X_train_resampled = np.expand_dims(X_train_resampled, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Instantiate the new MLP model
    model = MLP(num_features=num_features)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model, optimizer, criterion,
        X_train_resampled, y_train_resampled,
        X_val, y_val,
        epochs=30, batch_size=32, device='cpu'
    )

    acc, f1, rec, cm, precision = evaluate_model(model, X_test, y_test, device='cpu')
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")

    # Plot and save the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix\n Testing set with MLP')
    plt.savefig('../../plots/nn_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
