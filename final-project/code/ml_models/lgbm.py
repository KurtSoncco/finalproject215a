import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_val_test_split, process_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import os

if __name__ == "__main__":
    sns.set_palette("colorblind")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv('../../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target, val_size=0.2, test_size=0.2, random_state=42
    )
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # search space for hyperparameters
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 4, 5, 6]
    }

    model = LGBMClassifier(objective='binary', verbose=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    # evaluating the best model
    best_model = grid_search.best_estimator_
    best_model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_val, y_val)],
        eval_metric='f1',
    )

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
    plt.title('LightGBM', fontsize=16)
    plt.savefig('../../plots/lgbm_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    feature_names = df.columns[3:]
    fig, ax = plt.subplots(figsize=(50, 30))
    lgb.plot_tree(best_model, tree_index=0, ax=ax, orientation='vertical')
    plt.savefig("../../plots/lgbm_tree_vert.pdf")
    plt.close()