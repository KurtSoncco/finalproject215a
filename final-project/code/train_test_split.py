import pandas as pd
import numpy as np
import os

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
    
    return train_df, val_df, test_df, train_target, val_target, test_target