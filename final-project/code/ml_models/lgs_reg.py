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

# Import logistic regression
from sklearn.linear_model import LogisticRegression

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

    # Parameter grid for logistic regression
    # C is the regularization strength (inverse of regularization), larger C means weaker regularization.
    # Penalty choices for scikit-learn logistic regression typically 'l1', 'l2'; using 'liblinear' or 'saga' solver.
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # 'liblinear' supports both l1 and l2 penalties
    }

    # Initialize LogisticRegression
    model = LogisticRegression(max_iter=1000)

    # Use GridSearchCV with scoring='f1' to find best hyperparameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    # For logistic regression, we don’t have early stopping in the same sense.
    # We can just use the best model from GridSearchCV directly:
    best_model.fit(X_train_resampled, y_train_resampled)

    y_pred = best_model.predict(X_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Logistic regression doesn't have a tree to visualize.
    # If you need model interpretation, you could inspect coefficients:
    # print("Coefficients:", best_model.coef_)
