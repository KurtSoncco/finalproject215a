import os
import pandas as pd

# Define the path to the folder containing the CSV files
data_folder = 'data/CSpine/CSV datasets'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Initialize an empty dataframe
merged_df = pd.DataFrame()

# Iterate through each CSV file and merge them
for csv_file in csv_files:
    file_path = os.path.join(data_folder, csv_file)
    df = pd.read_csv(file_path)
    
    # Ensure all column names are in lowercase for case-insensitivity
    df.columns = df.columns.str.lower()
    
    # Merge using the common columns
    if merged_df.empty:
        merged_df = df
    else:
        merged_df = pd.merge(
            merged_df, 
            df, 
            on=["site", "caseid", "controltype", "studysubjectid"], 
            how="outer",
            suffixes=('', '_dup')  # Avoid default '_x', '_y' conflicts
        )
        # Remove duplicate columns by prioritizing the first dataset's columns
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)

print("Merged CSV saved as 'merged_data.csv'.")
