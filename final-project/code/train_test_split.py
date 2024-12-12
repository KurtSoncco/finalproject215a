import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

def site_train_val_test_split(df, target_df, val_size=0.2, test_size=0.2, random_state=42):
    """Split the data into train, validation, and test sets based on the SITE column.

    Args:
        df (pd.DataFrame): The input data.
        target_df (pd.DataFrame): The target data.
        val_size (float): The proportion of the data to include in the validation set.
        test_size (float): The proportion of the data to include in the test set.
        random_state (int): The random seed to use for reproducibility.
    
    Returns:
        tuple: A tuple containing the train, validation, and test sets for the input and target data.
    
    """

    # Extract number of unique sites
    n_sites = df.site.nunique()
    
    # Set random seed
    np.random.seed(random_state)
    
    # Select validation sites randomly
    val_sites = np.random.choice(df.site.unique(), size=int(n_sites * val_size), replace=False)
    
    # Select test sites randomly from remaining sites
    remaining_sites = df.site.unique()[~np.isin(df.site.unique(), val_sites)]
    test_sites = np.random.choice(remaining_sites, size=int(n_sites * test_size), replace=False)
    
    # Split the data
    val_df = df[df.site.isin(val_sites)]
    test_df = df[df.site.isin(test_sites)]
    train_df = df[~df.site.isin(np.concatenate((val_sites, test_sites)))]
    
    # Split the target
    val_target = target_df[df.site.isin(val_sites)]
    test_target = target_df[df.site.isin(test_sites)]
    train_target = target_df[~df.site.isin(np.concatenate((val_sites, test_sites)))]

    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state+1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state+2).reset_index(drop=True)
    
    # Drop columns
    if "site" in train_df.columns:
        train_df = train_df.drop(columns=["site"])
        val_df = val_df.drop(columns=["site"])
        test_df = test_df.drop(columns=["site"])

    if "caseid" in train_df.columns:
        train_df = train_df.drop(columns=["caseid"])
        val_df = val_df.drop(columns=["caseid"])
        test_df = test_df.drop(columns=["caseid"])

    if "studysubjectid" in train_df.columns:
        train_df = train_df.drop(columns=["studysubjectid"])
        val_df = val_df.drop(columns=["studysubjectid"])
        test_df = test_df.drop(columns=["studysubjectid"])
    
    train_target = train_target.drop(columns=["studysubjectid"])
    val_target = val_target.drop(columns=["studysubjectid"])
    test_target = test_target.drop(columns=["studysubjectid"])
    
    # Convert to numpy
    train_df = train_df.to_numpy()
    val_df = val_df.to_numpy()
    test_df = test_df.to_numpy()
    
    # Convert to numpy and ravel to 1D array
    train_target = train_target.values.ravel()
    val_target = val_target.values.ravel()
    test_target = test_target.values.ravel()
    
    # Add normalization for input features
    train_df = train_df.astype(np.float32)
    val_df = val_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # Normalize input features to [0,1] range
    def normalize_features(data):
        return np.clip((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8), 0, 1)

    train_df = normalize_features(train_df)
    val_df = normalize_features(val_df)
    test_df = normalize_features(test_df)

    # Convert targets to float32
    train_target = train_target.astype(np.float32)
    val_target = val_target.astype(np.float32)
    test_target = test_target.astype(np.float32)

    return train_df, val_df, test_df, train_target, val_target, test_target

def engineer_features(df):
    """Add engineered features based on research findings."""
    # Convert numpy array to DataFrame if necessary
    if isinstance(df, np.ndarray):
        return df  # For now, skip feature engineering for numpy arrays
        
    df = df.copy()
    
    # Define the columns we want to check for
    risk_columns = [
        'altered_mental_status',
        'focal_neurologic_findings',
        'neck_pain',
        'torticollis',
        'substantial_torso_injury',
        'predisposing_condition',
        'diving_injury',
        'high_risk_mvc'
    ]
    
    # Check if columns exist before creating composite features
    existing_columns = [col for col in risk_columns if col in df.columns]
    
    if existing_columns:
        # Create high risk factors only from existing columns
        risk_conditions = []
        for col in existing_columns:
            risk_conditions.append(df[col] == 1)
        
        if risk_conditions:
            df['high_risk_factors'] = (sum(risk_conditions) > 0).astype(int)
    
    # Add interaction terms only if both columns exist
    if 'altered_mental_status' in df.columns and 'focal_neurologic_findings' in df.columns:
        df['mental_status_neuro'] = df['altered_mental_status'] * df['focal_neurologic_findings']
    
    if 'neck_pain' in df.columns and 'torticollis' in df.columns:
        df['neck_symptoms'] = df['neck_pain'] * df['torticollis']
    
    return df.values if isinstance(df, pd.DataFrame) else df

def preprocess_data(X, is_training=True, imputer=None):
    """Preprocess data by handling NaN values."""
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        if is_training:
            X = imputer.fit_transform(X)
        else:
            X = imputer.transform(X)
        return X, imputer
    else:
        return imputer.transform(X), imputer

def balance_dataset(X_train, y_train, random_state=42):
    """Apply SMOTE + undersampling to balance the dataset."""
    # Ensure inputs are numpy arrays
    X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
    y_train = y_train if isinstance(y_train, np.ndarray) else y_train.values
    
    # Handle NaN values
    X_train, imputer = preprocess_data(X_train, is_training=True)
    
    sampling_pipeline = Pipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('undersampling', RandomUnderSampler(random_state=random_state))
    ])
    
    X_resampled, y_resampled = sampling_pipeline.fit_resample(X_train, y_train)
    return X_resampled, y_resampled, imputer

def select_features(X_train, y_train, X_val, X_test, max_features=20, random_state=42):
    """Select most important features using LightGBM."""
    # Ensure inputs are numpy arrays
    X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
    X_val = X_val if isinstance(X_val, np.ndarray) else X_val.values
    X_test = X_test if isinstance(X_test, np.ndarray) else X_test.values
    y_train = y_train if isinstance(y_train, np.ndarray) else y_train.values
    
    selector = SelectFromModel(
        LGBMClassifier(random_state=random_state),
        max_features=max_features
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_val_selected, X_test_selected

def enhanced_train_val_test_split(df, target_df, val_size=0.2, test_size=0.2, random_state=42):
    """Enhanced version of split that includes new preprocessing steps."""
    # First apply the original split
    train_df, val_df, test_df, train_target, val_target, test_target = site_train_val_test_split(
        df, target_df, val_size, test_size, random_state
    )
    
    # Skip feature engineering if data is already numpy array
    if isinstance(train_df, pd.DataFrame):
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)
    
    # Balance training data and get imputer
    train_df, train_target, imputer = balance_dataset(train_df, train_target, random_state)
    
    # Apply same imputation to validation and test sets
    val_df, _ = preprocess_data(val_df, is_training=False, imputer=imputer)
    test_df, _ = preprocess_data(test_df, is_training=False, imputer=imputer)
    
    # Select features
    train_df, val_df, test_df = select_features(
        train_df, train_target, val_df, test_df
    )
    
    return train_df, val_df, test_df, train_target, val_target, test_target