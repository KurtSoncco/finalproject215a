import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# XGBoost import
import xgboost as xgb

# import graphviz  # Only needed if you're planning on visualizing XGBoost trees with graphviz

def process_data(df):
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    df = df.rename(columns={"csfractures": "csi"})
    target = df["csi"]
    df.drop(columns=["csi"], inplace=True)
    target.replace(-1, 0, inplace=True)
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    df = df.select_dtypes(exclude=['object'])
    return df, target

if __name__ == "__main__":
    sns.set_palette("colorblind")
    df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target, val_size=0.2, test_size=0.2, random_state=42
    )
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100],        # analogous to CatBoost "iterations"
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 4, 5, 6]     # analogous to CatBoost "depth"
    }

    # Initialize XGBoost Classifier
    # eval_metric can be changed for the final fit (e.g., 'logloss' or 'error'), 
    # but for GridSearchCV scoring='f1', it’s enough to just provide the classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',  # Default internal metric for training
        verbosity=1
    )

    # Use GridSearchCV with scoring='f1' to find best parameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    # Now fit the best model on the training set again, using validation set for early stopping
    best_model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    y_pred = best_model.predict(X_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # XGBoost tree plot (optional)
    # If you'd like to plot a single tree:
    # xgb.plot_tree(best_model, num_trees=0)
    # plt.savefig('xgboost_tree.png')
    # plt.show()
