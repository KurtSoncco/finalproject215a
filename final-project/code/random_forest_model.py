# %% [markdown]
# # Random Forest Model for Cervical Spine Injury Prediction

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, make_scorer, roc_curve, auc
from train_test_split import site_train_val_test_split

# Set style for visualizations
sns.set_palette("colorblind")
plt.style.use('seaborn-v0_8')

def print_metrics(y_true, y_pred, title=""):
    """Print and plot comprehensive metrics with clinical focus."""
    if title:
        print(f"\n{title}")
    
    # Calculate confusion matrix for FNR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)  # False Negative Rate
    
    metrics = {
        'Sensitivity (Recall)': recall_score(y_true, y_pred, average='weighted'),
        'Specificity': recall_score(y_true, y_pred, pos_label=0, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred),
        'False Negative Rate': fnr
    }
    
    print('\nMetrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return metrics

def prepare_data():
    """Load and prepare data following Kurt's approach."""
    # Load data
    df = pd.read_csv('./data/merged_data_cleaned.csv', low_memory=False)
    target = pd.read_csv('./data/target.csv')
    target.columns = target.columns.str.lower()
    target.rename(columns={"csfractures": "csi"}, inplace=True)
    
    # Data cleaning following Kurt's approach
    df = df.select_dtypes(exclude=['object'])
    df = df.replace(-1, 0)
    df = df.drop(columns=['sectiongcsavailable'])
    
    # Reorder target to match df
    target = target.set_index('studysubjectid').loc[df['studysubjectid']].reset_index()
    
    # Add print statements to inspect data
    print(f"Shape of data before feature selection: {df.shape}")
    
    # Check for potential data leakage columns
    print("\nChecking for suspicious columns that might cause data leakage:")
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['fracture', 'injury', 'csi', 'diagnosis', 'outcome']):
            print(f"Potential leakage column: {col}")
    
    # More conservative feature dropping
    columns_to_drop = [
        "caseid", 
        "studysubjectid", 
        "ageinyears",
        # Add any columns identified as potential leakage
    ]
    
    df = df.drop(columns=columns_to_drop)
    print(f"\nShape of data after feature selection: {df.shape}")
    
    return df, target

def main():
    # Prepare data
    df, target = prepare_data()
    
    # Store column names before splitting
    feature_names = df.columns.tolist()  # Convert to list for consistency
    
    # Split data
    train_df, val_df, test_df, train_target, val_target, test_target = site_train_val_test_split(
        df, target['csi'], random_state=19
    )
    
    # Initialize Random Forest with more conservative parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    # Train model
    rf_model.fit(train_df, train_target)
    
    # Add debug prints AFTER model training
    print(f"Number of features: {len(feature_names)}")
    print(f"Length of feature importances: {len(rf_model.feature_importances_)}")
    
    # Make sure lengths match before creating DataFrame
    if len(feature_names) == len(rf_model.feature_importances_):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Most Important Features for Clinical Decision Making')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("Error: Number of features doesn't match feature importances")
        print("Features that might have been dropped:")
        print(set(feature_names) - set(df.columns))
    
    # Make predictions
    train_pred = rf_model.predict(train_df)
    val_pred = rf_model.predict(val_df)
    test_pred = rf_model.predict(test_df)
    
    # Evaluate model
    print_metrics(train_target, train_pred, "Training Set")
    print_metrics(val_target, val_pred, "Validation Set")
    print_metrics(test_target, test_pred, "Test Set")

if __name__ == "__main__":
    main()