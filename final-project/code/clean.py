import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

def merging_datasets():
# Define the path to the folder containing the CSV files
    data_folder = '../data/CSpine/CSV datasets'

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
    merged_df.to_csv('../data/merged_data.csv', index=False)

    print("Merged CSV saved as 'merged_data.csv'.")

# Cleaning by Finn
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
    
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{title.lower().replace(' ', '_')}.png")
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
            plt.savefig("../plots/review_results_distribution.png")
            plt.close()
        
        if 'reviewresultnumofviews' in df_cleaned.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df_cleaned, x='reviewresultnumofviews', bins=20)
            plt.title('Distribution of Number of Views')
            plt.tight_layout()
            plt.savefig("../plots/number_of_views_distribution.png")
            plt.close()
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

# Cleaning by Kurt

# Analysis variables
def clean_analysis(df_analysis):
    """Clean the analysis dataset."""
    
    # Fill missing values with 0
    df_analysis.fillna(0, inplace=True)

    return df_analysis


# Clinical on Field

def clean_onfield(df_field):
    """ Clean the on field dataset."""
    # Change "YES" to 1 and everything else to 0
    df_field.loc[:, "pttenderneckothertxtcat"] = df_field["pttenderneckothertxtcat"].apply(lambda x: 1 if x == "YES" else 0)
    df_field.loc[:, "otherneurodeficitdesccat"] = df_field["otherneurodeficitdesccat"].apply(lambda x: 1 if x == "YES" else 0)

    # Change "ALTERED" to 1 and everything else to 0
    df_field.loc[:, "avpumentaltxtcat"] = df_field["avpumentaltxtcat"].apply(lambda x: 1 if x == "ALTERED" else 0)

    # If avpumental is U, change tthe column of avpudetails to U
    df_field.loc[df_field["avpumental"] == "U", "avpudetails"] = "U"

    # Change "OTH" to 1 and everything else to 0
    df_field.loc[:, "avpumental"] = df_field["avpumental"].apply(lambda x: 1 if x == "OTH" else 0)

    # Eliminate all columns with txt at end
    df_field = df_field.loc[:, ~df_field.columns.str.endswith("txt")]
    df_field = df_field.loc[:, ~df_field.columns.str.endswith("desc")]

    # Since we are interested on loss of consciousness, we will join the manual and total GCS values
    df_field.loc[df_field["totalgcsavailable"] =="Y", "totalgcs"] = df_field[df_field["totalgcsavailable"] =="Y"]["totalgcsmanual"]
    df_field.drop(columns=["totalgcsmanual", "sectiongcsavailable", "totalgcsavailable", "gcseye", "verbalgcs", "motorgcs"], inplace=True)

    return df_field


# Clinical Outside

def clean_outside(df_outside):
    """ Clean the outside dataset."""
    # Change "YES" to 1 and everything else to 0
    df_outside.loc[:, "pttenderneckothertxtcat"] = df_outside["pttenderneckothertxtcat"].apply(lambda x: 1 if x == "YES" else 0)
    df_outside.loc[:, "otherneurodeficitdesccat"] = df_outside["otherneurodeficitdesccat"].apply(lambda x: 1 if x == "YES" else 0)

    # Change "ALTERED" to 1 and everything else to 0
    df_outside.loc[:, "avpumentaltxtcat"] = df_outside["avpumentaltxtcat"].apply(lambda x: 1 if x == "ALTERED" else 0)

    # If avpumental is U, change tthe column of avpudetails to U
    df_outside.loc[df_outside["avpumental"] == "U", "avpudetails"] = "U"

    # Change "OTH" to 1 and everything else to 0
    df_outside.loc[:, "avpumental"] = df_outside["avpumental"].apply(lambda x: 1 if x == "OTH" else 0)


    # Eliminate all columns with txt at end
    df_outside = df_outside.loc[:, ~df_outside.columns.str.endswith("txt")]
    df_outside = df_outside.loc[:, ~df_outside.columns.str.endswith("desc")]

    # Since we are interested on loss of consciousness, we will join the manual and total GCS values
    #df_outside.loc[df_outside["totalgcsavailable"] =="Y", "totalgcs"] = df_outside[df_outside["totalgcsavailable"] =="Y"]["totalgcsmanual"]
    #df_outside.drop(columns=["totalgcsmanual", "sectiongcsavailable", "totalgcsavailable", "gcseye", "verbalgcs", "motorgcs"], inplace=True)

    return df_outside


# Creating the target data

def create_target(df_analysis, df_injuryclassification):

    # Convert the columns of CSFractures to True and False
    A = df_injuryclassification["CSFractures"].apply(lambda x: True if x == "Y" else False)
    B = df_injuryclassification["Ligamentoptions"].apply(lambda x: True if x == "Y" else False)

    # Create a new column with the logical OR of the two columns
    df_injuryclassification["CSI"] = A | B

    # Merge the two dataframes on the column "StudySubjectID"
    df_target = pd.merge(df_analysis, df_injuryclassification, on="StudySubjectID", how="left")

    # Only consider the columns "StudySubjectID" and "CSI"
    df_target = df_target[["StudySubjectID", "CSI"]]

    # Fill the NaN values with False
    df_target["CSFractures"] = df_target["CSFractures"].fillna(False)

    # Rename the column "CSFractures" to "CSI"
    df_target.rename(columns={"CSFractures": "CSI"}, inplace=True)

    # Export the dataframe to a CSV file
    df_target.to_csv("../data/target_data.csv", index=False)

    return df_target

# cleaning by jacob

def standardize(df):
    # Make sure every dataset has the same column "StudySubjectID"
    if "StudySubjectID" not in df.columns:
        if "studysubjectid" in df.columns.str.lower():
            df.rename(columns={"studysubjectid": "StudySubjectID"}, inplace=True)
            # Change the type to int
            df["StudySubjectID"] = df["StudySubjectID"].astype(int)

    # Also check "SITE","CaseID","ControlType"
    if "SITE" not in df.columns:
        print("SITE not in columns")
        print(df.columns)
        if "site" in df.columns.str.lower():
            print("SITE is in columns but with different case")
            df.rename(columns={"site": "SITE"}, inplace=True)
            df["SITE"] = df["SITE"].astype(int)
    if "CaseID" not in df.columns:
        print("CaseID not in columns")
        print(df.columns)
        if "caseid" in df.columns.str.lower():
            print("CaseID is in columns but with different case")
            df.rename(columns={"caseid": "CaseID"}, inplace=True)
            df["CaseID"] = df["CaseID"].astype(int)
    if "ControlType" not in df.columns:
        print("ControlType not in columns")
        print(df.columns)
        if "controltype" in df.columns.str.lower():
            print("ControlType is in columns but with different case")
            df.rename(columns={"controltype": "ControlType"}, inplace=True)
    return df

def binarize(df):
    for cols in df.columns:
        if 'Y' in list(df[cols].unique()) and 'N' in list(df[cols].unique()):
            print(cols)
            df[cols] = df[cols].replace('Y', 1)
            df[cols] = df[cols].replace('N', 0)
    return df


def clean_clinical_site(clinical):
    # threshold for dropping columns with excessive missing data
    threshold = 80.0
    
    # drop columns w/ missing above threshold
    clinical_cleaned = clinical.loc[:, (clinical.isnull().mean() * 100) <= threshold]

    df = clinical_cleaned.drop(columns=["arrivaldate","arrivaltime", "arrivaltimend", "modearrival"])

    df = binarize(df)

    # GCSEye
    eye_mapping = {
            1 : np.nan,
            2 : "Pain",
            3 : "Verbal",
            4 : "Spontaneous"
    }
    #df.GCSEye = df.GCSEye.map(eye_mapping)
    
    # VerbalGCS
    verbal_mapping = {
            1 : np.nan,
            2 : "Incomprehensible sounds - moans",
            3 : "Inappropriate words - cries to pain",
            4 : "Confused - irritable/cries",
            5 : "Oriented - coos/babbles"
    }
    #df.VerbalGCS = df.VerbalGCS.map(verbal_mapping)
    
    # LocEvalPhysician
    loc_mapping = {
            1 : "ED",
            2 : "ICU",
            3 : "General floor",
            4 : "Outpatient Clinic",
            5 : "Other"
    }
    df.locevalphysician = df.locevalphysician.map(loc_mapping) 
    
    # CervicalSpineImmobilization
    immobilization_mapping = {
            1 : 1,
            2 : np.nan,
            3 : 0
    }
    df.cervicalspineimmobilization = df.cervicalspineimmobilization.map(immobilization_mapping)
        
    # IntubatedSS
    intubated_mapping = {
            "Y" : 1,
            "NOTUB" : 0,
            "INTUB" : "Intubation continued",
            "EXTUB" : "Extubated"
    }
    df.intubatedss = df.intubatedss.map(intubated_mapping)
        
    # CSpinePrecautions
    cspine_mapping = {
            "YD" : 1,
            "N" : 0,
            "YND" : np.nan
    }
    df.cspineprecautions = df.cspineprecautions.map(cspine_mapping)
    
    # PtSensoryLoss
    sensory_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "ND" : np.nan
    }
    df.ptsensoryloss = df.ptsensoryloss.map(sensory_mapping)
        
    # PtParesthesias
    paresthesias_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "ND" : np.nan
    }
    df.ptparesthesias = df.ptparesthesias.map(paresthesias_mapping)
        
    # LimitedRangeMotion
    range_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "4" : "C-collar in place",
            "NA" : "NA",
            "ND" : np.nan
    }
    df.limitedrangemotion = df.limitedrangemotion.map(range_mapping)    
    
    # MotorGCS
    motor_mapping = {
            1 : np.nan,
            2 : "Abnormal extension posturing",
            3 : "Abnormal flexure posturing",
            4 : "Withdraws to pain",
            5 : "Localizes pain [withdraws to touch]",
            6 : "Follow Commands"
    } 
    #df.MotorGCS = df.MotorGCS.map(motor_mapping)

    return df

def clean_demo(demo):
    # threshold for dropping columns with excessive missing data
    threshold = 80.0
    
    # drop columns w/ missing above threshold
    demo_cleaned = demo.loc[:, (demo.isnull().mean() * 100) <= threshold]

    demo_cleaned = binarize(demo_cleaned)

    return demo_cleaned

def clean_injuryclass(injury_class):
    # threshold for dropping columns with excessive missing data
    threshold = 80.0
    
    # drop columns w/ missing above threshold
    injury_class_cleaned = injury_class.loc[:, (injury_class.isnull().mean() * 100) <= threshold]

    injury_class_cleaned = binarize(injury_class_cleaned)

    return injury_class_cleaned

def main():
    """
    Main function to execute the cleaning process
    """
    try:
        print("Reading data...")
        # Merge the data
        merging_datasets()

        # Read the merged data
        df_total = pd.read_csv("../data/merged_data.csv", low_memory=False)
        df_total.columns = df_total.columns.str.lower()
        
        print("\nInitial data shape:", df_total.shape)
        print("\nColumns with missing values:")
        print(df_total.isnull().sum()[df_total.isnull().sum() > 0])
        
        print("\nCleaning data...")
        df_cleaned = clean_radiology_variables(df_total)
        df_cleaned = clean_analysis(df_cleaned)
        df_cleaned = clean_onfield(df_cleaned)
        df_cleaned = clean_outside(df_cleaned)
        df_cleaned = clean_clinical_site(df_cleaned)
        df_cleaned = clean_demo(df_cleaned)

        # If specified, drop 80% missing columns
        drop_missing = False
        if drop_missing:
            df_cleaned = clean_demo(df_cleaned)
        
        # Verify the cleaning process
        verify_cleaning(df_total, df_cleaned)
        
        os.makedirs("../data", exist_ok=True)
        df_cleaned.to_csv("../data/merged_data_cleaned.csv", index=False)
        print("\nData cleaning completed successfully.")
        print("\nData shape after cleaning:", df_cleaned.shape)
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

if __name__ == "__main__":
    main()
    


