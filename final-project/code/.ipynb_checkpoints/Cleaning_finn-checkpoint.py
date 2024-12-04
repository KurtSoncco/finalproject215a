import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("colorblind")

def clean_binary_yn(series: pd.Series) -> pd.Series:
    """
    Convert Y/N values to 1/0, handling missing values and unexpected inputs
    """
    mapping = {'Y': 1, 'N': 0}
    cleaned = pd.to_numeric(series.map(mapping), errors='coerce')
    return cleaned

def clean_radiology_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all radiology-related variables in the merged dataset
    """
    df_cleaned = df.copy()
    
    # 1. Binary Y/N Variables
    binary_columns = [
        'xrays', 'ctperformed', 'mriperformed', 
        'writtenordictatedconsult', 'operativereport'
    ]
    
    for col in binary_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = clean_binary_yn(df_cleaned[col])
            
    # 2. XraysView Variables - already binary but ensure numeric
    xray_view_columns = [
        'xraysviewap', 'xraysviewlt', 'xraysviewom',
        'xraysviewfe', 'xraysviewsw', 'xraysviewot'
    ]
    
    for col in xray_view_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 3. Review Result Variables
    if 'reviewresultnumofviews' in df_cleaned.columns:
        df_cleaned['reviewresultnumofviews'] = pd.to_numeric(
            df_cleaned['reviewresultnumofviews'], 
            errors='coerce'
        )

    # Main review result - categorical
    result_mapping = {
        'AT': 'Abnormal Technical',
        'EQ': 'Equivocal',
        'NC': 'Non Contributory',
        'NM': 'Normal',
        'ND': np.nan
    }
    
    if 'reviewresult' in df_cleaned.columns:
        df_cleaned['reviewresult'] = df_cleaned['reviewresult'].map(result_mapping)

    # 4. Review Result Detail Columns
    detail_columns = [
        'reviewresultat_df', 'reviewresultat_li', 'reviewresultat_oa',
        'reviewresulteq_pf', 'reviewresulteq_ll', 'reviewresulteq_as',
        'reviewresulteq_li', 'reviewresulteq_of'
    ]
    
    for col in detail_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 5. Special Review Results
    iv_mapping = {
        'IVN': 0,
        'IVY': 1,
        'IVND': np.nan
    }
    
    if 'reviewresultiv' in df_cleaned.columns:
        df_cleaned['reviewresultiv'] = df_cleaned['reviewresultiv'].map(iv_mapping)

    can_mapping = {
        'CAN': 1,
        'CAND': np.nan
    }
    
    if 'reviewresultca' in df_cleaned.columns:
        df_cleaned['reviewresultca'] = df_cleaned['reviewresultca'].map(can_mapping)

    # 6. Outside ED Variables
    outside_ed_mapping = {
        'SITE': 'Study Site',
        'EDA': 'Outside ED Attended',
        'EDNA': 'Outside ED Not Attended'
    }
    
    if 'outsideed' in df_cleaned.columns:
        df_cleaned['outsideed'] = df_cleaned['outsideed'].map(outside_ed_mapping)

    return df_cleaned

def plot_binary_distributions(df: pd.DataFrame, columns: list, title: str) -> None:
    """
    Plot distribution of binary variables before and after cleaning
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*4, 4))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        if col in df.columns:
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_ylabel('Count')
        else:
            print(f"Warning: Column {col} not found in data")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs("final-project/plots", exist_ok=True)
    plt.savefig(f"final-project/plots/{title.lower().replace(' ', '_')}.png")
    plt.close()

def verify_cleaning(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> None:
    """
    Verify the cleaning process with various checks and visualizations
    """
    print("\nData Cleaning Verification:")
    print("-" * 50)
    
    # 1. Check for missing values
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])
    
    # 2. Value distributions for categorical variables
    print("\nValue distributions for key categorical variables:")
    categorical_cols = ['reviewresult', 'outsideed']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            print(f"\n{col}:")
            print(df_cleaned[col].value_counts(dropna=False))
        else:
            print(f"Warning: Column {col} not found in data")
    
    # 3. Binary variables verification
    binary_cols = [
        'xrays', 'ctperformed', 'mriperformed', 
        'writtenordictatedconsult', 'operativereport'
    ]
    print("\nBinary variables unique values:")
    for col in binary_cols:
        if col in df_cleaned.columns:
            print(f"{col}: {sorted(df_cleaned[col].unique())}")
        else:
            print(f"Warning: Column {col} not found in data")
    
    # 4. Visualizations
    try:
        plot_binary_distributions(
            df_cleaned, 
            binary_cols,
            "Distribution of Binary Variables After Cleaning"
        )
        
        if 'reviewresult' in df_cleaned.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_cleaned, x='reviewresult')
            plt.title('Distribution of Review Results')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("final-project/plots/review_results_distribution.png")
            plt.close()
        
        if 'reviewresultnumofviews' in df_cleaned.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df_cleaned, x='reviewresultnumofviews', bins=20)
            plt.title('Distribution of Number of Views')
            plt.tight_layout()
            plt.savefig("final-project/plots/number_of_views_distribution.png")
            plt.close()
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

def main():
    """
    Main function to execute the cleaning process
    """
    try:
        print("Reading data...")
        df_total = pd.read_csv("final-project/data/merged_data.csv", low_memory=False)
        df_total.columns = df_total.columns.str.lower()
        
        print("\nInitial data shape:", df_total.shape)
        print("\nColumns with missing values:")
        print(df_total.isnull().sum()[df_total.isnull().sum() > 0])
        
        print("\nCleaning data...")
        df_cleaned = clean_radiology_variables(df_total)
        
        verify_cleaning(df_total, df_cleaned)
        
        os.makedirs("final-project/data", exist_ok=True)
        df_cleaned.to_csv("final-project/data/merged_data_cleaned.csv", index=False)
        print("\nData cleaning completed successfully.")
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

if __name__ == "__main__":
    main()



