import subprocess

def run_file(file_name):
    if file_name.endswith('.py'):
        subprocess.run(['python', file_name])
    elif file_name.endswith('.ipynb'):
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', file_name])

files_to_run = [
    'EDA.py',
    'random_forest_model.py',
    'tbfn.py'
    'nn.py',
    'clean_RF_features.py',
    'ml_models/lgbm.py',
    'ml_models/cb.py',
    'ml_models/xgb.py',
    'ml_models/lgs_reg.py',
    'ml_models/linear.py',
    'ml_models/random_forest.py',
    'nn_models/nn.py',
    'nn_models/1dcnn.py',
    'nn_models/tbfn.py',
]

for file in files_to_run:
    run_file(file)