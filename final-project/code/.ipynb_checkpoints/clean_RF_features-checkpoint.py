import pandas as pd
import numpy as np

# Paths to your data files
merged_data_path = './data/merged_data_cleaned.csv'
target_path = './data/target.csv'
output_path = './data/cleaned_eight_factors.csv'

df = pd.read_csv(merged_data_path, low_memory=False)
target = pd.read_csv(target_path)

# Convert True/False in target to 0/1
target['csi'] = target['CSFractures'].astype(int)

# Merge target onto df based on StudySubjectID and studysubjectid
df_merged = pd.merge(df, target[['StudySubjectID', 'csi']], left_on='studysubjectid', right_on='StudySubjectID', how='inner')

# Include the site column and the eight factors plus csi
# Adjust column names if they differ:
factor_columns = {
    'altered_mental_status': 'alteredmentalstatus',
    'focal_neurologic_findings': 'focalneurofindings',
    'neck_pain': 'painneck',
    'torticollis': 'torticollis',
    'predisposing_condition': 'predisposed',
    'substantial_torso_injury': 'subinj_torsotrunk',
    'diving': 'highriskdiving',
    'high_risk_mvc': 'highriskmvc'
}

selected_cols = ['site'] + list(factor_columns.values()) + ['csi', 'studysubjectid']
df_selected = df_merged[selected_cols]

# Rename columns for clarity
df_selected.rename(columns={
    factor_columns['altered_mental_status']: 'altered_mental_status',
    factor_columns['focal_neurologic_findings']: 'focal_neurologic_findings',
    factor_columns['neck_pain']: 'neck_pain',
    factor_columns['torticollis']: 'torticollis',
    factor_columns['predisposing_condition']: 'predisposing_condition',
    factor_columns['substantial_torso_injury']: 'substantial_torso_injury',
    factor_columns['diving']: 'diving',
    factor_columns['high_risk_mvc']: 'high_risk_mvc'
}, inplace=True)

# Replace any -1 with 0 if needed
df_selected = df_selected.replace(-1, 0)

# Ensure binary cols are int
binary_cols = [
    'altered_mental_status', 'focal_neurologic_findings', 'neck_pain',
    'torticollis', 'predisposing_condition', 'substantial_torso_injury',
    'diving', 'high_risk_mvc', 'csi'
]
for col in binary_cols:
    df_selected[col] = df_selected[col].astype(int)

# Save final CSV
df_selected.to_csv(output_path, index=False)
print("cleaned_eight_factors.csv created successfully with 'site' column!")
