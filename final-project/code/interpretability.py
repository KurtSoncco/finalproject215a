import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

sns.set_palette("colorblind")

def decision_curve_analysis(y_true, y_proba, thresholds):
    net_benefit = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        benefit = tp - fp * (threshold / (1 - threshold))
        net_benefit.append(benefit / len(y_true))
    return net_benefit

def plot_decision_curve(net_benefit):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit, label='Model')
    plt.plot(thresholds, [0] * len(thresholds), label='Treat None', linestyle='--')
    plt.plot(thresholds, [sum(y_test) / len(y_test) - (1 - sum(y_test) / len(y_test)) * (threshold / (1 - threshold)) for threshold in thresholds], label='Treat All', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.savefig('../plots/tabpfn_decision_curve.pdf', bbox_inches='tight', dpi=300)
    plt.show()

def process_data(df, target):
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    df = df.select_dtypes(exclude=['object'])
    return df, target

if __name__ == '__main__':
    # Read data
    df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)
    target = pd.read_csv('../data/target.csv')
    target.columns = target.columns.str.lower()

    target.rename(columns={"csfractures": "csi"}, inplace=True)

    # Process data
    df, target = process_data(df, target)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)

    ## Use SMOTE to handle class imbalance
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
    # Calculate the Specificity and Sensitivity
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"Specificity: {specificity}")
    print(f"Sensitivity: {sensitivity}")

    # False Negative Rate
    fnr = 1 - sensitivity
    print(f"False Negative Rate: {fnr}")

    # Decision curve analysis
    thresholds = np.linspace(0, 0.25, 100)
    y_proba = model.predict_proba(X_test)[:, 1]
    net_benefit = decision_curve_analysis(y_test, y_proba, thresholds)
    plot_decision_curve(net_benefit)


    