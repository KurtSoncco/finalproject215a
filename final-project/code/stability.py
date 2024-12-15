import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def sampling_site_with_less_prevalence(df, target_df, n_sites=5):

    """Stability 1: Training on a differnt site covariance """

    # Get the proportion of true values per site in the target dataset
    
    ## Add site column from df to target_df
    target_df['site'] = df['site'].values

    ## Calculate the proportion of true values per site
    site_proportion = target_df.groupby('site').mean().drop(columns=["studysubjectid"])

    #print(f"Site proportion: {site_proportion}")

    ## Get the top 5 sites with the least proportion of true values
    site_proportion = site_proportion.sort_values(by='csi', ascending=True)
    sites = site_proportion.index[:n_sites] ## This sites would be the testing sites
    #print(f"Testing sites: {sites}")

    ## Get the indices of the sites in the df dataset
    site_indices = df['site'].values

    ## Get the indices of the sites in the target dataset that are not in the sites list
    train_indices = [i for i, site in enumerate(site_indices) if site not in sites]

    ## Get the indices of the sites in the target dataset that are in the sites list
    test_indices = [i for i, site in enumerate(site_indices) if site in sites]

    ## Get the training and testing datasets
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    train_target = target_df.iloc[train_indices]
    test_target = target_df.iloc[test_indices]

    ## Drop the site column from the target_df
    for col in ['site','caseid','studysubjectid']:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col], errors='ignore')
            test_df = test_df.drop(columns=[col], errors='ignore')
    if 'studysubjectid' in train_target.columns:
        train_target = train_target.drop(columns=["studysubjectid"], errors='ignore')
        test_target = test_target.drop(columns=["studysubjectid"], errors='ignore')
    if 'site' in train_target.columns:
        train_target = train_target.drop(columns=["site"], errors='ignore')
        test_target = test_target.drop(columns=["site"], errors='ignore')

    # Check if the sum of the training and testing datasets is equal to the original dataset
    assert train_df.shape[0] + test_df.shape[0] == df.shape[0], "The sum of the training and testing datasets should be equal to the original dataset"
    assert train_target.shape[0] + test_target.shape[0] == target_df.shape[0], "The sum of the training and testing datasets should be equal to the original dataset"

    # Check that target only has one column
    assert train_target.shape[1] == 1, "The target dataset should only have one column"

    # Get the values
    X_train = train_df.to_numpy().astype(np.float32)
    X_test = test_df.to_numpy().astype(np.float32)
    y_train = train_target.values.ravel().astype(np.float32)
    y_test = test_target.values.ravel().astype(np.float32)

    # Check if the shapes are correct
    assert X_train.shape[0] == y_train.shape[0], "The number of samples in the training dataset should be equal to the number of targets"
    assert X_test.shape[0] == y_test.shape[0], "The number of samples in the testing dataset should be equal to the number of targets"

    return X_train, X_test, y_train, y_test



def subsampling_train_test_split(df, target_df, subsample_ratio = 0.5):
    """ Subsample for the same site, and do multiple subsampling ratios """

    # Extract a random site
    site = 1 # 11

    # Get the indices of the site in the target dataset
    data = df[df['site'] != site]
    target = target_df[df['site'] != site]

    # Drop identifying columns
    for col in ['site', 'caseid', 'studysubjectid']:
        if col in data.columns:
            data = data.drop(columns=[col], errors='ignore')
    if 'studysubjectid' in target.columns:
        target = target.drop(columns=["studysubjectid"], errors='ignore')
    if 'site' in target.columns:
        target = target.drop(columns=["site"], errors='ignore')

    # Subsample the data
    n_samples = int(len(data) * subsample_ratio)
    trainingsamples = data.sample(n=n_samples)
    trainingtarget = target.loc[trainingsamples.index]

    # Get the indices of the training samples
    train_indices = trainingsamples.index

    assert data.shape[0] == target.shape[0], "The number of samples in the data should be equal to the number"
    print(f"Number of samples: {train_indices.shape[0]}")

    # Testing data is outside the training data
    test_indices = [i for i in data.index if i not in train_indices]
    test_data = data.loc[test_indices]
    test_target = target.loc[test_indices]

    # Get training and testing datasets
    X_train, X_test, y_train, y_test = trainingsamples.to_numpy().astype(np.float32), test_data.to_numpy().astype(np.float32), trainingtarget.values.ravel().astype(np.float32), test_target.values.ravel().astype(np.float32)


    assert X_train.shape[0] == y_train.shape[0], "The number of samples in the training dataset should be equal to the number of targets"
    assert X_test.shape[0] == y_test.shape[0], "The number of samples in the testing dataset should be equal to the number of targets"

    return X_train, X_test, y_train, y_test








