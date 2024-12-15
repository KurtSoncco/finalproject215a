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
    data['target'] = target_df.values.ravel()

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

<<<<<<< Updated upstream
if __name__ == "__main__":
    
    # Assuming df_model has columns:
    # 'site', 'altered_mental_status', 'focal_neurologic_findings', 'neck_pain', 'torticollis',
    # 'predisposing_condition', 'substantial_torso_injury', 'diving', 'high_risk_mvc', 'csi'
    df_model = pd.read_csv("./data/cleaned_eight_factors.csv")

    # Separate features and target
    X = df_model[['altered_mental_status',
                'focal_neurologic_findings',
                'neck_pain',
                'torticollis',
                'predisposing_condition',
                'substantial_torso_injury',
                'diving',
                'high_risk_mvc']].values
    y = df_model['csi'].values
    X_train, X_val, X_test, y_train, y_val, y_test = site_train_val_test_split(df_model, df_model[['studysubjectid','csi']], 
                                                                            val_size=0.2, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(rf, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # Instead of using the training set for threshold determination, use the validation set:
    y_val_proba = best_rf.predict_proba(X_val)[:, 1]

    best_sensitivity = 0.0
    best_threshold = 0.5
    for t in np.linspace(0, 1, 101):
        y_pred_t = (y_val_proba >= t).astype(int)
        sensitivity_t = recall_score(y_val, y_pred_t)
        # Pick threshold that gives max sensitivity on validation set
        if sensitivity_t > best_sensitivity:
            best_sensitivity = sensitivity_t
            best_threshold = t

    print("Best threshold for maximum sensitivity on validation data:", best_threshold)
    print("Sensitivity at this threshold on validation:", best_sensitivity)

    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    auc_test = roc_auc_score(y_test, y_test_proba)
    fnr_test = fn / (fn + tp)  # False Negative Rate

    print("Test Sensitivity:", sensitivity_test)
    print("Test Specificity:", specificity_test)
    print("Test AUC:", auc_test)
    print("Test FNR:", fnr_test)
=======
>>>>>>> Stashed changes
