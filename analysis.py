import subprocess

def run_file(file_name):
    if file_name.endswith('.py'):
        subprocess.run(['python', file_name])
    elif file_name.endswith('.ipynb'):
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', file_name])

files_to_run = [
    'logistic_regression.py',
    'neural_network.py',
    'random_forest_model.py',
    'jacob_ml.ipynb'
]

for file in files_to_run:
    run_file(file)