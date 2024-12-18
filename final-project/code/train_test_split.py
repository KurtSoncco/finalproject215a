import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# We'll assume df_model still has 'site' column; if not, you'll need to re-merge or include it.
# The following function splits data based on site:
def site_train_val_test_split(df, target_df, val_size=0.2, test_size=0.2, random_state=42):
    """Site-based split into train, val, and test."""
    n_sites = df.site.nunique()
    np.random.seed(random_state)
    val_sites = np.random.choice(df.site.unique(), size=int(n_sites * val_size), replace=False)
    remaining_sites = df.site.unique()[~np.isin(df.site.unique(), val_sites)]
    test_sites = np.random.choice(remaining_sites, size=int(n_sites * test_size), replace=False)


    # Split data based on sites
    val_df = df[df.site.isin(val_sites)]
    test_df = df[df.site.isin(test_sites)]
    train_df = df[~df.site.isin(np.concatenate((val_sites, test_sites)))]

    # Split target data based on sites
    val_target = target_df[df.site.isin(val_sites)]
    test_target = target_df[df.site.isin(test_sites)]
    train_target = target_df[~df.site.isin(np.concatenate((val_sites, test_sites)))]
    
    # Shuffle the data
    np.random.seed(random_state)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state+1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state+2).reset_index(drop=True)
    
    # Drop identifying columns
    for col in ['site','caseid','studysubjectid']:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col], errors='ignore')
            val_df = val_df.drop(columns=[col], errors='ignore')
            test_df = test_df.drop(columns=[col], errors='ignore')
    if 'studysubjectid' in train_target.columns:
        train_target = train_target.drop(columns=["studysubjectid"], errors='ignore')
        val_target = val_target.drop(columns=["studysubjectid"], errors='ignore')
        test_target = test_target.drop(columns=["studysubjectid"], errors='ignore')
    
    X_train = train_df.to_numpy().astype(np.float32)
    X_val = val_df.to_numpy().astype(np.float32)
    X_test = test_df.to_numpy().astype(np.float32)
    y_train = train_target.values.ravel().astype(np.float32)
    y_val = val_target.values.ravel().astype(np.float32)
    y_test = test_target.values.ravel().astype(np.float32)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_test_split(df, target_df, val_size=0.2, test_size=0.2, random_state=42):
    """Randomly split data into train, val, and test sets."""
    # Combine features and target
    data = df.copy()
    data['target'] = target_df['csi'].values.ravel()

    # Drop identifying columns
    for col in ['site', 'caseid', 'studysubjectid']:
        if col in data.columns:
            data = data.drop(columns=[col], errors='ignore')

    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data.drop('target', axis=1), data['target'], test_size=test_size, random_state=random_state
    )

    # Adjust val_size for the remaining data
    val_size_adjusted = val_size / (1 - test_size)

    # Split train+val into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )

    # Convert to numpy arrays
    X_train = X_train.to_numpy().astype(np.float32)
    X_val = X_val.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test

