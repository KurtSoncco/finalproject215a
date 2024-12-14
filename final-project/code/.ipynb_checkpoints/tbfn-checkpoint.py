import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from train_test_split import train_val_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier

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
    # Apply PCA to reduce the number of features to 150
    pca = PCA(n_components=100)
    X_train_resampled = pca.fit_transform(X_train_resampled)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    # Initialize TabPFNClassifier
    model = TabPFNClassifier(device='cpu')
    model.fit(X_train_resampled, y_train_resampled, overwrite_warning=True)

    # Predict on the test data
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
