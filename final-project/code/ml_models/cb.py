import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_val_test_split, process_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sns.set_palette("colorblind")
    df = pd.read_csv('../../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target, val_size=0.2, test_size=0.2, random_state=42)
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    param_grid = {
        'iterations': [100],
        'learning_rate': [0.001, 0.01, 0.1],
        'depth': [3, 4, 5, 6]
    }

    # hyperparameter sweep with CatBoost initialized with default parameters
    model = CatBoostClassifier(loss_function='Logloss', eval_metric='F1', verbose=200)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled, eval_set=(X_val, y_val), early_stopping_rounds=100)

    y_pred = best_model.predict(X_test)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('CatBoost', fontsize=16)
    plt.savefig('../../plots/cb_confusion_matrix.pdf', dpi=300, bbox_inches='tight')