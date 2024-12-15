import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LPCA import LogisticPCA

# Set colorblind palette
sns.set_palette("colorblind")

# Metrics function
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score

def print_metrics(y_true, y_pred):
    """ Print various metrics for a classification model. 
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
    Returns:
        dict: Dictionary of metrics
    """

    metrics = {
        'F1 score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'Accuracy': balanced_accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred)
    }

    # Print metrics
    print('Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    # Confusion matrix
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    return metrics


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('./data/merged_data_cleaned.csv', low_memory=False)
    target = pd.read_csv('./data/target.csv')
    target.columns = target.columns.str.lower()
    target.rename(columns={"csfractures": "csi"}, inplace=True)

    # Feature preparation - do this BEFORE splitting
    df_processed = df.drop(columns=["caseid", "studysubjectid", "arrivaltime", "controltype"])
    df_processed = df_processed.select_dtypes(exclude=['object'])  # Remove non-numeric columns
    df_processed = pd.get_dummies(df_processed, drop_first=True)  # One-hot encode remaining columns

    # Only select columns that contains 0 and 1
    df_processed = df_processed.loc[:, (df_processed.nunique() == 2)]
    df_processed = df_processed.fillna(0)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = site_train_val_test_split(df_processed, target)

    # Train logistic regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate model
    print_metrics(y_val, y_pred)

    # LPCA

    # Check the best number of components
    k_values = range(1, 25)
    metrics_list = []

    for k in k_values:
        lpca = LogisticPCA(m=6, k=k, verbose=False)
        lpca.fit(df_processed.values, target["csi"])
        transformed_X = lpca.transform(X.values)
        X_new = lpca.sigmoid(transformed_X)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_new, target["csi"])
        pred = model.predict(X_new)
        metrics = print_metrics(target["csi"], pred)
        metrics_list.append(metrics)

    # Select all the F1, accuracy and recall score
    f1_scores = [metrics['F1 score'] for metrics in metrics_list]
    accuracy_scores = [metrics['Accuracy'] for metrics in metrics_list]
    recall_scores = [metrics['Recall'] for metrics in metrics_list]

    # Plot #1 : Number of components
    plt.plot(k_values, f1_scores, 'x-', color="blue")
    plt.plot(k_values, accuracy_scores, 'x-', color="red")
    plt.plot(k_values, recall_scores, 'x-', color="green")
    plt.xticks(k_values)
    plt.xlabel('Number of components')
    plt.ylabel('Score')
    plt.title('LPCA\nMetrics vs Number of components')
    plt.legend(['F1 score', 'Accuracy', 'Recall'])
    plt.ylim([0.5, 1])
    plt.grid()
    plt.savefig('../plots/lpca_metrics_vs_k.pdf', dpi=300, bbox_inches='tight')

    # Re run for validation set and test set
    df_processed["site"] = df["site"]

    # Split data
    train_df, val_df, test_df, train_target, val_target, test_target = site_train_val_test_split(df_processed, target)

    # LPCA for Validation set
    k_values = range(1, 25)
    metrics_list = []

    for k in k_values:
        lpca = LogisticPCA(m=6, k=k, verbose=False)
        lpca.fit(val_df, maxiters=50, tol=1e-5)
        transformed_X = lpca.transform(val_df)
        X_new = lpca.sigmoid(transformed_X)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_new, val_target)
        pred = model.predict(X_new)
        metrics = print_metrics(val_target, pred)
        metrics_list.append(metrics)

    # Select all the F1, accuracy and recall score
    f1_scores = [metrics['F1 score'] for metrics in metrics_list]
    accuracy_scores = [metrics['Accuracy'] for metrics in metrics_list]
    recall_scores = [metrics['Recall'] for metrics in metrics_list]

    # Plot #2 : Number of components for validation set
    plt.plot(k_values, f1_scores, 'x-', color="blue")
    plt.plot(k_values, accuracy_scores, 'x-', color="red")
    plt.plot(k_values, recall_scores, 'x-', color="green")
    plt.xticks(k_values) # Set the x-ticks to be the k values
    plt.xlabel('Number of components')
    plt.ylabel('Score')
    plt.title('LPCA\n Validation Metrics vs Number of components')
    plt.legend(['F1 score', 'Accuracy', 'Recall'])
    plt.ylim([None, 1])
    plt.grid()
    plt.savefig('../plots/lpca_metrics_vs_k_val.pdf', dpi=300, bbox_inches='tight')

    # LPCA for Test set
    k_values = range(1, 25)
    metrics_list = []

    for k in k_values:
        lpca = LogisticPCA(m=6, k=k, verbose=False)
        lpca.fit(test_df, maxiters=50, tol=1e-5)
        transformed_X = lpca.transform(test_df)
        X_new = lpca.sigmoid(transformed_X)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_new, test_target)
        pred = model.predict(X_new)
        metrics = print_metrics(test_target, pred)
        metrics_list.append(metrics)

    # Select all the F1, accuracy and recall score
    f1_scores = [metrics['F1 score'] for metrics in metrics_list]
    accuracy_scores = [metrics['Accuracy'] for metrics in metrics_list]
    recall_scores = [metrics['Recall'] for metrics in metrics_list]

    # Plot #3 : Number of components for test set
    plt.plot(k_values, f1_scores, 'x-', color="blue")
    plt.plot(k_values, accuracy_scores, 'x-', color="red")
    plt.plot(k_values, recall_scores, 'x-', color="green")
    plt.xticks(k_values) # Set the x-ticks to be the k values
    plt.xlabel('Number of components')
    plt.ylabel('Score')
    plt.title('LPCA\nTesting Metrics vs Number of components')
    plt.legend(['F1 score', 'Accuracy', 'Recall'])
    plt.ylim([None, 1])
    plt.grid()
    plt.savefig('../plots/lpca_metrics_vs_k_test.pdf', dpi=300, bbox_inches='tight')


    # Confusion matrix on testing set
    sns.heatmap(confusion_matrix(test_target, pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix\n Testing set with LPCA')
    plt.savefig('../plots/lpca_confusion_matrix.pdf', dpi=300, bbox_inches='tight')


