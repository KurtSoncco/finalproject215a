import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from train_test_split import site_train_val_test_split, enhanced_train_val_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(8)
np.random.seed(8)

def prepare_data():
    # Load data
    df = pd.read_csv('./data/merged_data_cleaned.csv', low_memory=False)
    target = pd.read_csv('./data/target.csv')
    target.columns = target.columns.str.lower()
    target.rename(columns={"csfractures": "csi"}, inplace=True)
    
    # Feature preparation - do this BEFORE splitting
    df_processed = df.drop(columns=["caseid", "studysubjectid", "arrivaltime", "controltype"])
    df_processed = df_processed.select_dtypes(exclude=['object'])  # Remove non-numeric columns
    df_processed = pd.get_dummies(df_processed, drop_first=True)  # One-hot encode remaining columns
    
    # Use enhanced split with processed dataframe
    train_df, val_df, test_df, train_target, val_target, test_target = enhanced_train_val_test_split(
        df_processed, target
    )
    
    return train_df, val_df, test_df, train_target, val_target, test_target

class CSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)  # Ensure target is correct shape
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CSpineNet(nn.Module):
    def __init__(self, input_dim):
        super(CSpineNet, self).__init__()
        # Much simpler architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Changed to Kaiming initialization
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class CSpineDataset(Dataset):
    def __init__(self, features, targets):
        # Convert features to tensor, handling both DataFrame and numpy array inputs
        self.features = torch.FloatTensor(features if isinstance(features, np.ndarray) else features.values)
        # Convert targets to tensor, handling both Series/DataFrame and numpy array inputs
        self.targets = torch.FloatTensor(targets if isinstance(targets, np.ndarray) else targets.values)
        
        # Add normalization here
        self.features = (self.features - self.features.min(dim=0)[0]) / (self.features.max(dim=0)[0] - self.features.min(dim=0)[0] + 1e-8)
        self.features = torch.nan_to_num(self.features, 0.0)  # Replace NaN with 0
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            targets = targets.float().view(-1, 1)
            # Calculate weighted loss and reduce to scalar
            loss = criterion(outputs, targets)
            if isinstance(loss, torch.Tensor) and len(loss.shape) > 0:
                loss = loss.mean()  # Ensure loss is scalar
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                targets = targets.float().view(-1, 1)
                outputs = model(features)
                loss = criterion(outputs, targets)
                if isinstance(loss, torch.Tensor) and len(loss.shape) > 0:
                    loss = loss.mean()  # Ensure loss is scalar
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return best_model, train_losses, val_losses

def evaluate_model(model, data_loader, device, threshold=0.5):  # Back to standard threshold
    model.eval()
    all_preds = []
    all_targets = []
    all_raw_outputs = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            preds = (outputs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_raw_outputs.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate confusion matrix for FNR
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fnr = fn / (fn + tp)  # False Negative Rate
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = float('nan')
    
    return accuracy, f1, auc, fnr

def main():
    # Get data using prepare_data function
    train_df, val_df, test_df, train_target, val_target, test_target = prepare_data()
    
    # Create datasets
    train_dataset = CSpineDataset(train_df, train_target)
    val_dataset = CSpineDataset(val_df, val_target)
    test_dataset = CSpineDataset(test_df, test_target)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSpineNet(input_dim=train_df.shape[1]).to(device)
    
    # Moderate class weight
    pos_weight = torch.tensor([(1 - train_target.mean()) / train_target.mean() * 2.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Use Adam with moderate parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # Less aggressive learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Larger batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Train model with fewer epochs
    best_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, num_epochs=100
    )
    
    # Load best model and evaluate
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Evaluate on all sets
    print("\nFinal Evaluation:")
    for name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
        accuracy, f1, auc, fnr = evaluate_model(model, loader, device)
        print(f"{name} Set Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  False Negative Rate: {fnr:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 