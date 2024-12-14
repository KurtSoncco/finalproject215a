import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# Assume df_model now includes 'site' and the eight factors plus 'csi'
df_model = pd.read_csv("./data/cleaned_eight_factors.csv")

# Extract features and target
X = df_model[['altered_mental_status',
              'focal_neurologic_findings',
              'neck_pain',
              'torticollis',
              'predisposing_condition',
              'substantial_torso_injury',
              'diving',
              'high_risk_mvc']].values
y = df_model['csi'].values

# site_train_val_test_split function defined previously to split based on site
from train_test_split import site_train_val_test_split

X_train, X_val, X_test, y_train, y_val, y_test = site_train_val_test_split(df_model, df_model[['studysubjectid','csi']], 
                                                                          val_size=0.2, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(rf, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Get validation predictions (probabilities)
y_val_proba = best_rf.predict_proba(X_val)[:, 1]

# We want at least 90% sensitivity but not at the cost of zero specificity.
desired_sensitivity = 0.9
best_threshold = None
best_f1 = -1.0

thresholds = np.linspace(0, 1, 101)
for t in thresholds:
    y_val_pred_t = (y_val_proba >= t).astype(int)
    sens_t = recall_score(y_val, y_val_pred_t)
    if sens_t >= desired_sensitivity:
        prec_t = precision_score(y_val, y_val_pred_t, zero_division=0)
        f1_t = f1_score(y_val, y_val_pred_t)
        # Choose the threshold that yields the best F1-score among those that meet the sensitivity criterion
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

print("Chosen threshold that achieves at least 90% sensitivity and maximizes F1 on validation data:", best_threshold)

# Evaluate on the test set using this chosen threshold
y_test_proba = best_rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn + fp)
test_precision = tp / (tp + fp) if (tp+fp) > 0 else 0
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)
test_fnr = fn / (fn + tp) if (fn+tp) > 0 else 0

print("Test Sensitivity:", test_sensitivity)
print("Test Specificity:", test_specificity)
print("Test Precision:", test_precision)
print("Test F1 Score:", test_f1)
print("Test AUC:", test_auc)
print("Test FNR:", test_fnr)
