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

# Import linear regression
from sklearn.linear_model import LinearRegression

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

    # Parameter grid for linear regression (few hyperparameters)
    # 'fit_intercept' is commonly tuned. You could also consider 'normalize' or 'positive',
    # but let's keep it simple.
    param_grid = {
        'fit_intercept': [True, False]
    }

    model = LinearRegression()

    # GridSearchCV with scoring='f1' is unusual for linear regression,
    # but we'll treat the predictions as binary via a 0.5 threshold after the predictions.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',  # uses binary classification metric on thresholded regression output
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled)

    # Predict continuous values
    y_pred_continuous = best_model.predict(X_test)
    # Convert continuous values into binary predictions using a threshold of 0.5
    y_pred = (y_pred_continuous >= 0.5).astype(int)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Linear regression doesn't have a tree structure to plot.
    # If you want to inspect the coefficients:
    # print("Coefficients:", best_model.coef_)
    # print("Intercept:", best_model.intercept_)
