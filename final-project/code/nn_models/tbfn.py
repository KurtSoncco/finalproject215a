import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_val_test_split, process_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier
import os

if __name__ == "__main__":
    sns.set_palette("colorblind")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv('../../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    pca_model = PCA(n_components=100)
    X_train_resampled = pca_model.fit_transform(X_train_resampled)
    X_test = pca_model.transform(X_test)
    # Initialize TabPFNClassifier
    model = TabPFNClassifier(device="cpu")
    model.fit(X_train_resampled, y_train_resampled, overwrite_warning=True)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix\n Testing set with TabPFN')
    plt.savefig('../../plots/tbfn_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
