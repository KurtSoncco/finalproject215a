import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from train_test_split import train_val_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTENC, SVMSMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier
from stability import sampling_site_with_less_prevalence, subsampling_train_test_split


sns.set_palette("colorblind")

def process_data(df, target):
    df.drop(columns=["injurydatetime", "arrivaldate", "arrivaltime"], inplace=True)
    #target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()
    df = pd.get_dummies(df, columns=["controltype"], drop_first=True)
    df = df.select_dtypes(exclude=['object'])
    return df, target


if __name__ == "__main__":
    # Read data
    df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)
    target = pd.read_csv('../data/target.csv')
    target.columns = target.columns.str.lower()

    target.rename(columns={"csfractures": "csi"}, inplace=True)

    # Process data
    df, target = process_data(df, target)

    # Stability 1
    stab1 = False
    if stab1:
        # Sampling site with less prevalence
        accuracy_stab1 = []
        f1_score_stab1 = []
        recall_stab1 = []
        specificity_stab1 = []
        sensitivity_stab1 = []
        fnr_stab1 = []

        for i in range(1,6):
            X_train, X_test, y_train, y_test = sampling_site_with_less_prevalence(df, target, n_sites=i)

            ## Use SMOTE to handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            # Apply PCA to reduce the number of features to 150
            pca = PCA(n_components=100)
            X_train_resampled = pca.fit_transform(X_train_resampled)
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

            # Calculate the FNR
            fnr = fn / (fn + tp)
            print(f"FNR: {fnr}")

            # Append the metrics to the lists
            accuracy_stab1.append(accuracy_score(y_test, y_pred))
            f1_score_stab1.append(f1_score(y_test, y_pred))
            recall_stab1.append(recall_score(y_test, y_pred))
            specificity_stab1.append(specificity)
            sensitivity_stab1.append(sensitivity)
            fnr_stab1.append(fnr)

        # Plot of stability 1 metrics
        plt.figure(figsize=(8, 6))
        plt.plot(range(1,6), f1_score_stab1, label='F1 Score')
        plt.plot(range(1,6), specificity_stab1, label='Specificity')
        plt.plot(range(1,6), sensitivity_stab1, label='Sensitivity')
        plt.plot(range(1,6), fnr_stab1, label='FNR')
        plt.xlabel('Number of sites used for testing')
        plt.ylabel('Metrics')
        plt.title('Stability 1 Metrics')
        plt.legend()
        plt.savefig('../plots/tabpfn_stability1_metrics.pdf', bbox_inches='tight', dpi=300)

    ## Stability 2
    ## Subsampling train test split

    subsampling_ratios = [0.5, 0.6, 0.7, 0.8]

    accuracy_stab2 = []
    f1_score_stab2 = []
    recall_stab2 = []
    specificity_stab2 = []
    sensitivity_stab2 = []
    fnr_stab2 = []

    stab2 = False

    if stab2:
        for ratio in subsampling_ratios:
            X_train, X_test, y_train, y_test = subsampling_train_test_split(df, target, subsample_ratio = ratio)

            ## Use SMOTE to handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            # Apply PCA to reduce the number of features to 100
            pca = PCA(n_components=100)
            X_train_resampled = pca.fit_transform(X_train_resampled)
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

            # Calculate the FNR
            fnr = fn / (fn + tp)
            print(f"FNR: {fnr}")

            # Append the metrics to the lists
            accuracy_stab2.append(accuracy_score(y_test, y_pred))
            f1_score_stab2.append(f1_score(y_test, y_pred))
            recall_stab2.append(recall_score(y_test, y_pred))
            specificity_stab2.append(specificity)
            sensitivity_stab2.append(sensitivity)
            fnr_stab2.append(fnr)

        # Plot of stability 2 metrics
        plt.figure(figsize=(8, 6))
        plt.plot(subsampling_ratios, f1_score_stab2, label='F1 Score')
        plt.plot(subsampling_ratios, specificity_stab2, label='Specificity')
        plt.plot(subsampling_ratios, sensitivity_stab2, label='Sensitivity')
        plt.plot(subsampling_ratios, fnr_stab2, label='FNR')
        plt.xlabel('Subsampling Ratios')
        plt.ylabel('Metrics')
        plt.title('Stability 2 Metrics')
        plt.legend()
        plt.savefig('../plots/tabpfn_stability2_metrics.pdf', bbox_inches='tight', dpi=300)


    # Stability 3: SMOTE techniques

    smote_techniques = [SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE]

    accuracy_stab3 = []
    f1_score_stab3 = []
    recall_stab3 = []
    specificity_stab3 = []
    sensitivity_stab3 = []
    fnr_stab3 = []

    # Eliminate SubjectID
    target.drop(columns=['studysubjectid'], inplace=True)

    # Ravel the target
    target = target.values.ravel()

    for smote_technique in smote_techniques:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)


        # Use the current SMOTE technique to handle class imbalance
        smote = smote_technique(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        # Apply PCA to reduce the number of features to 100
        pca = PCA(n_components=100)
        X_train_resampled = pca.fit_transform(X_train_resampled)
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

        # Calculate the FNR
        fnr = fn / (fn + tp)
        print(f"FNR: {fnr}")

        # Append the metrics to the lists
        accuracy_stab3.append(accuracy_score(y_test, y_pred))
        f1_score_stab3.append(f1_score(y_test, y_pred))
        recall_stab3.append(recall_score(y_test, y_pred))
        specificity_stab3.append(specificity)
        sensitivity_stab3.append(sensitivity)
        fnr_stab3.append(fnr)

        # Plot confusion matrix
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        #plt.xlabel('Predicted')
        #plt.ylabel('Actual')
        #plt.title(f'Confusion Matrix for {smote_technique.__name__}')
        #plt.savefig(f'../plots/tabpfn_confusion_matrix_stab3_{smote_technique.__name__}.pdf', bbox_inches='tight', dpi=300)

    # Plot of stability 3 metrics
    plt.figure(figsize=(8, 6))
    plt.plot([tech.__name__ for tech in smote_techniques], f1_score_stab3, label='F1 Score')
    plt.plot([tech.__name__ for tech in smote_techniques], specificity_stab3, label='Specificity')
    plt.plot([tech.__name__ for tech in smote_techniques], sensitivity_stab3, label='Sensitivity')
    plt.plot([tech.__name__ for tech in smote_techniques], fnr_stab3, label='FNR')
    plt.xlabel('SMOTE Techniques')
    plt.ylabel('Metrics')
    plt.title('Stability 3 Metrics')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('../plots/tabpfn_stability3_metrics.pdf', bbox_inches='tight', dpi=300)