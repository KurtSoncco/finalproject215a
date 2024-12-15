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

# Use RandomForestClassifier instead of CatBoost or LGBM
from sklearn.ensemble import RandomForestClassifier

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
    df = pd.read_csv('../../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target, val_size=0.2, test_size=0.2, random_state=42
    )
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = RandomForestClassifier(random_state=42)

    # GridSearchCV with scoring='f1'
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    # Re-fit on the entire training set (RandomForest doesn't have the same 'early_stopping' approach)
    best_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Optional: visualize one of the trees in the Random Forest
    # from sklearn.tree import plot_tree
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20, 10))
    # plot_tree(best_model.estimators_[0], filled=True, feature_names=df.columns)
    # plt.savefig("random_forest_tree.png")
    # plt.show()
