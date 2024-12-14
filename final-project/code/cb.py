import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils import resample
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sns.set_palette("colorblind")
df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)
target = pd.read_csv('../data/target.csv')
target.columns = target.columns.str.lower()

target.rename(columns={"csfractures": "csi"}, inplace=True)
target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()

def process_data(df, target):
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    df = df.select_dtypes(exclude=['object'])
    return df, target

if __name__ == "__main__":
    df, target = process_data(df, target)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    param_grid = {
        'iterations': [100],
        'learning_rate': [0.01, 0.001, 0.1],
        'depth': [3, 4, 5, 6]
    }
    # Apply PCA to reduce the number of features to 100
    pca = PCA(n_components=150)
    X_train_resampled = pca.fit_transform(X_train_resampled)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    model = CatBoostClassifier(loss_function='Logloss', eval_metric='F1', verbose=200)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled, eval_set=(X_val, y_val), early_stopping_rounds=100)

    y_pred = best_model.predict(X_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()