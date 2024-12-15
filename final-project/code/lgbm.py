import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split import site_train_val_test_split, train_val_test_split
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import resample
# Replace CatBoost imports with LightGBM
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
# Remove catboost-specific imports (Pool, CatBoost, etc.)
# import graphviz  # You can keep this if you need Graphviz for LGBM tree visualization

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
    df = pd.read_csv('../data/merged_data_cleaned.csv', low_memory=False)

    df, target = process_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target, val_size=0.2, test_size=0.2, random_state=42
    )
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Adjust parameter names to match LightGBM
    param_grid = {
        'n_estimators': [100],               # replaces 'iterations'
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 4, 5, 6]            # replaces 'depth'
    }

    model = LGBMClassifier(objective='binary', verbose=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    # LightGBM supports an eval_set & early stopping inside fit, but GridSearchCV alone won't use early stopping
    # directly. The final best model can be fit again with early stopping as shown below.
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    # Now fit the best model on the training set with early stopping on the validation set
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
    print("Confusion Matrix:")
    print(cm)

    # If you want a tree plot for LightGBM, you can use:
    fig, ax = plt.subplots(figsize=(50, 30))
    lgb.plot_tree(best_model, tree_index=0, ax=ax)
    plt.savefig("lgbm_tree.png")

    # The CatBoost-specific code for plotting the first tree is commented out
    # since LightGBM uses a different plotting interface:
    # pool = Pool(X_train_resampled, y_train_resampled, feature_names=list(df.columns[3:]))
    # plot_model = CatBoostClassifier(loss_function='Logloss', eval_metric='F1', verbose=200, **grid_search.best_params_)
    # plot_model.fit(pool)
    # fig = plot_model.plot_tree(tree_idx=0)
    # fig.render("catboost_tree")
