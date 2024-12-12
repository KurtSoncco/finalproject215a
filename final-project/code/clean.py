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
    df_analysis.fillna(-1, inplace=True)
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

def clean_kappa(kappa):
    kappa["ptambulatorypriorarrival"].replace('3', 2, inplace=True)
    kappa["ptambulatorypriorarrival"].replace('ND', 2, inplace=True)
    kappa["ptambulatorypriorarrival"].replace('N', 0, inplace=True)
    kappa["ptambulatorypriorarrival"].replace('Y', 1, inplace=True)
    kappa["hxlocsite"].replace('N', 0, inplace=True)
    kappa["hxlocsite"].replace('Y', 1, inplace=True)
    kappa["hxlocsite"].replace('ND', 2, inplace=True)
    kappa["hxlocsite"].replace('3', 2, inplace=True)
    kappa["hxlocsite"].replace('U', 2, inplace=True)
    kappa["sectiongcsavailable"].replace('Y', 1, inplace=True)
    kappa["sectiongcsavailable"].replace('ND', 2, inplace=True)
    kappa["sectiongcsavailable"].replace('Y', 1, inplace=True)
    kappa["sectiongcsavailable"].replace('ND', 2, inplace=True)
    kappa["sectiongcsavailable"] = kappa["sectiongcsavailable"].astype(int)
    kappa["totalgcsmanual"].fillna(-1, inplace=True)
    kappa["totalgcsmanual"] = kappa["totalgcsmanual"].astype(int)
    kappa["gcseye"].fillna(-1, inplace=True)
    kappa["totalgcs"].fillna(-1, inplace=True)
    kappa["motorgcs"].fillna(-1, inplace=True)
    kappa["avpu"].replace('N', 0, inplace=True)
    kappa["avpu"].replace('Y', 1, inplace=True)
    kappa["avpudetails"].replace('A', 0, inplace=True)
    kappa["avpudetails"].replace('P', 1, inplace=True)
    kappa["avpudetails"].replace('V', 2, inplace=True)
    kappa["avpudetails"].replace('U', 3, inplace=True)
    kappa["avpudetails"].replace('N', 4, inplace=True)
    kappa["avpudetails"].fillna(-1, inplace=True)
    kappa["avpumental"].replace("OTH", 0, inplace=True)
    kappa["avpumental"].fillna(-1, inplace=True)
    kappa["avpumentaltxtcat"].replace('UNALTERED', 0, inplace=True)
    kappa["avpumentaltxtcat"].replace('ALTERED', 1, inplace=True)
    kappa["avpumentaltxtcat"].fillna(-1, inplace=True)
    kappa["ptcomppain"].replace('N', 0, inplace=True)
    kappa["ptcomppain"].replace('Y', 1, inplace=True)
    kappa["ptcomppain"].fillna(-1, inplace=True)
    kappa["ptcomppain"].replace('ND', -1, inplace=True)
    kappa["ptcomppain"].replace('S', -1, inplace=True)
    kappa["ptcomppain"].replace('YND', -1, inplace=True)
    kappa["ptcomppain"].replace('P', -1, inplace=True)
    kappa["ptcomppainneckmove"].fillna(-1, inplace=True)
    kappa["ptcomppainneckmove"].replace('ND', -1, inplace=True)
    kappa["ptcomppainneckmove"].replace('N', 0, inplace=True)
    kappa["ptcomppainneckmove"].replace('Y', 1, inplace=True)
    kappa["pttender"].replace('N', 0, inplace=True)
    kappa["pttender"].replace('Y', 1, inplace=True)
    kappa["pttender"].fillna(-1, inplace=True)
    kappa["pttender"].replace('ND', -1, inplace=True)
    kappa["pttender"].replace('S', -1, inplace=True)
    kappa["pttenderneckothertxtcat"].replace('No', 0, inplace=True)
    kappa["pttenderneckothertxtcat"].replace('Yes', 1, inplace=True)
    kappa["pttenderneckothertxtcat"].fillna(-1, inplace=True)
    kappa["limitedrangemotion"].replace('N', 0, inplace=True)
    kappa["limitedrangemotion"].replace('Y', 1, inplace=True)
    kappa["limitedrangemotion"].fillna(-1, inplace=True)
    kappa["limitedrangemotion"].replace('ND', -1, inplace=True)
    kappa["limitedrangemotion"].replace('3', -1, inplace=True)
    kappa["limitedrangemotion"].replace('4', -1, inplace=True)
    kappa["otherinjuries"].replace('N', 0, inplace=True)
    kappa["otherinjuries"].replace('Y', 1, inplace=True)
    kappa["minorinjuries"].replace('N', 0, inplace=True)
    kappa["minorinjuries"].replace('Y', 1, inplace=True)
    kappa["ptparesthesias"].replace('N', 0, inplace=True)
    kappa["ptparesthesias"].replace('Y', 1, inplace=True)
    kappa["ptparesthesias"].fillna(-1, inplace=True)
    kappa["ptparesthesias"].replace('ND', -1, inplace=True)
    kappa["ptparesthesias"].replace('3', -1, inplace=True)
    kappa["ptsensoryloss"].replace('N', 0, inplace=True)
    kappa["ptsensoryloss"].replace('Y', 1, inplace=True)
    kappa["ptsensoryloss"].fillna(-1, inplace=True)
    kappa["ptsensoryloss"].replace('ND', -1, inplace=True)
    kappa["ptsensoryloss"].replace('3', -1, inplace=True)
    kappa["ptextremityweakness"].replace('N', 0, inplace=True)
    kappa["ptextremityweakness"].replace('Y', 1, inplace=True)
    kappa["ptextremityweakness"].fillna(-1, inplace=True)
    kappa["ptextremityweakness"].replace('ND', -1, inplace=True)
    kappa["ptextremityweakness"].replace('3', -1, inplace=True)
    kappa["otherneurodeficit"].replace('N', 0, inplace=True)
    kappa["otherneurodeficit"].replace('Y', 1, inplace=True)
    kappa["otherneurodeficit"].fillna(-1, inplace=True)
    kappa["otherneurodeficit"].replace('ND', -1, inplace=True)
    kappa["otherneurodeficitdesccat"].replace('No', 0, inplace=True)
    kappa["otherneurodeficitdesccat"].replace('Yes', 1, inplace=True)
    kappa["otherneurodeficitdesccat"].fillna(-1, inplace=True)
    kappa["intervforcervicalstab"].replace('N', 0, inplace=True)
    kappa["intervforcervicalstab"].replace('Y', 1, inplace=True)
    kappa["outcomestudysite"].replace('N', 0, inplace=True)
    kappa["outcomestudysite"].replace('PND', 1, inplace=True)
    kappa["outcomestudysite"].replace('DTH', 2, inplace=True)
    kappa["outcomestudysiteneuro"].replace('NR', 0, inplace=True)
    kappa["outcomestudysiteneuro"].replace('MD', 1, inplace=True)
    kappa["outcomestudysiteneuro"].replace('SD', 2, inplace=True)
    kappa["outcomestudysiteneuro"].replace('PVS', 3, inplace=True)
    kappa["outcomestudysiteneuro"].fillna(-1, inplace=True)
    kappa["outcomestudysitemobility"].replace('N', 0, inplace=True)
    kappa["outcomestudysitemobility"].replace('DA', 1, inplace=True)
    kappa["outcomestudysitemobility"].replace('WD', 2, inplace=True)
    kappa["outcomestudysitemobility"].replace('I', 3, inplace=True)
    kappa["outcomestudysitemobility"].fillna(-1, inplace=True)
    kappa["outcomestudysitebowel"].replace('N', 0, inplace=True)
    kappa["outcomestudysitebowel"].replace('I', 1, inplace=True)
    kappa["outcomestudysitebowel"].fillna(-1, inplace=True)
    kappa["outcomestudysiteurine"].replace('N', 0, inplace=True)
    kappa["outcomestudysiteurine"].replace('I', 1, inplace=True)
    kappa["outcomestudysiteurine"].replace('C', -1, inplace=True)
    kappa["outcomestudysiteurine"].fillna(-1, inplace=True)
    kappa["fielddocumentation"].fillna(-1, inplace=True)
    kappa["fielddocumentation"].replace('EMS', 0, inplace=True)
    kappa["fielddocumentation"].replace('NR', 1, inplace=True)
    kappa["fielddocumentation"].replace('OTR', 2, inplace=True)
    kappa["ptambulatoryprioremsarrival"].replace('N', 0, inplace=True)
    kappa["ptambulatoryprioremsarrival"].replace('Y', 1, inplace=True)
    kappa["ptambulatoryprioremsarrival"].fillna(-1, inplace=True)
    kappa["ptambulatoryprioremsarrival"].replace('ND', -1, inplace=True)
    kappa["ptambulatoryprioremsarrival"].replace('PA', -1, inplace=True)
    kappa["patientsposition"].fillna(-1, inplace=True)
    kappa["patientsposition"].replace('L', 0, inplace=True)
    kappa["patientsposition"].replace('ND', -1, inplace=True)
    kappa["patientsposition"].replace('S', 1, inplace=True)
    kappa["patientsposition"].replace('IDEMS', 2, inplace=True)
    kappa["patientsposition"].replace('S', 3, inplace=True)
    kappa["patientsposition"].replace('PA', 4, inplace=True)
    kappa["patientsposition"].replace('W', 5, inplace=True)
    kappa["hxlocfield"].fillna(-1, inplace=True)
    kappa["hxlocfield"].replace('N', 0, inplace=True)
    kappa["hxlocfield"].replace('Y', 1, inplace=True)
    kappa["hxlocfield"].replace('ND', -1, inplace=True)
    kappa["hxlocfield"].replace('U', -1, inplace=True)
    kappa["hxlocfield"].replace('S', -1, inplace=True)
    kappa["eddocumentation"].fillna(-1, inplace=True)
    kappa["eddocumentation"].replace('SITE', 0, inplace=True)
    kappa["eddocumentation"].replace('ED', 1, inplace=True)
    kappa["eddocumentation"].replace('EDU', 2, inplace=True)
    kappa["ptambulatorypriorarrivaled"].fillna(-1, inplace=True)
    kappa["ptambulatorypriorarrivaled"].replace('N', 0, inplace=True)
    kappa["ptambulatorypriorarrivaled"].replace('Y', 1, inplace=True)
    kappa["ptambulatorypriorarrivaled"].replace('ND', -1, inplace=True)
    kappa["hxloced"].fillna(-1, inplace=True)
    kappa["hxloced"].replace('N', 0, inplace=True)
    kappa["hxloced"].replace('Y', 1, inplace=True)
    kappa["hxloced"].replace('ND', -1, inplace=True)
    kappa["hxloced"].replace('U', -1, inplace=True)
    kappa["hxloced"].replace('S', -1, inplace=True)
    kappa["helmet"].replace('N', 0, inplace=True)
    kappa["helmet"].replace('Y', 1, inplace=True)
    kappa["helmet"].fillna(-1, inplace=True)
    kappa["helmet"].replace('ND', -1, inplace=True)
    kappa["clotheslining"].fillna(-1, inplace=True)
    kappa["clotheslining"].replace('N', 0, inplace=True)
    kappa["clotheslining"].replace('Y', 1, inplace=True)
    kappa["clotheslining"].replace('ND', -1, inplace=True)
    return kappa

def clean_medical_history(medicalhistory):
    medicalhistory["bodyasawhole"].replace('N', 0, inplace=True)
    medicalhistory["bodyasawhole"].replace('A', 1, inplace=True)
    medicalhistory["bodyasawhole"].fillna(-1, inplace=True)
    medicalhistory["heent"].replace('N', 0, inplace=True)
    medicalhistory["heent"].replace('A', 1, inplace=True)
    medicalhistory["heent"].fillna(-1, inplace=True)
    medicalhistory["cardiovascular"].replace('N', 0, inplace=True)
    medicalhistory["cardiovascular"].replace('A', 1, inplace=True)
    medicalhistory["cardiovascular"].fillna(-1, inplace=True)
    medicalhistory["respiratory"].fillna(-1, inplace=True)
    medicalhistory["respiratory"].replace('N', 0, inplace=True)
    medicalhistory["respiratory"].replace('A', 1, inplace=True)
    medicalhistory["gastrointestinal"].replace('N', 0, inplace=True)
    medicalhistory["gastrointestinal"].replace('A', 1, inplace=True)
    medicalhistory["gastrointestinal"].fillna(-1, inplace=True)
    medicalhistory["genitourinary"].replace('N', 0, inplace=True)
    medicalhistory["genitourinary"].replace('A', 1, inplace=True)
    medicalhistory["genitourinary"].fillna(-1, inplace=True)
    medicalhistory["musculoskeletal"].replace('N', 0, inplace=True)
    medicalhistory["musculoskeletal"].replace('A', 1, inplace=True)
    medicalhistory["musculoskeletal"].fillna(-1, inplace=True)
    medicalhistory["neurological"].fillna(-1, inplace=True)
    medicalhistory["neurological"].replace('N', 0, inplace=True)
    medicalhistory["neurological"].replace('A', 1, inplace=True)
    medicalhistory["endocrinological"].replace('N', 0, inplace=True)
    medicalhistory["endocrinological"].replace('A', 1, inplace=True)
    medicalhistory["endocrinological"].fillna(-1, inplace=True)
    medicalhistory["dermatologicalskin"].replace('N', 0, inplace=True)
    medicalhistory["dermatologicalskin"].replace('A', 1, inplace=True)
    medicalhistory["dermatologicalskin"].fillna(-1, inplace=True)
    medicalhistory["hematologiclymphatic"].replace('N', 0, inplace=True)
    medicalhistory["hematologiclymphatic"].replace('A', 1, inplace=True)
    medicalhistory["hematologiclymphatic"].fillna(-1, inplace=True)
    medicalhistory["medications"].replace('N', 0, inplace=True)
    medicalhistory["medications"].replace('Y', 1, inplace=True)
    medicalhistory["medications"].fillna(-1, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('Arnold Chiari Malformation', 0, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('Cervical spinal stenosis', 1, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('Achondrodysplasia', 2, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('Down\'s Syndrome', 3, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('Congenital anomaly of cervical vertebrae', 4, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('C-spine fusion, Larsen\'s Syndrome', 5, inplace=True)
    medicalhistory["otherpredisposingcondition"].replace('C-spine injury', 6, inplace=True)
    medicalhistory["otherpredisposingcondition"].fillna(-1, inplace=True)
    return medicalhistory

def clean_injury_mechanism(injurymechanism):
    injurymechanism["injurytime"] = pd.to_datetime(injurymechanism["injurytime"], format='%H:%M:%S', errors='coerce')
    injurymechanism["injurydate"] = pd.to_datetime(injurymechanism["injurydate"], format='%Y-%m-%d', errors='coerce')
    injurymechanism["injurydatetime"] = injurymechanism.apply(
        lambda row: pd.Timestamp.combine(row["injurydate"], row["injurytime"].time()) if pd.notnull(row["injurydate"]) and pd.notnull(row["injurytime"]) else pd.NaT,
        axis=1
    )
    injurymechanism["estimatetimeinjury"].replace('U', -1, inplace=True)
    injurymechanism["estimatetimeinjury"].fillna(-1, inplace=True)
    injurymechanism["injuryprimarymechanism"].replace('ND', -1, inplace=True)
    injurymechanism["clotheslining"].replace('N', 0, inplace=True)
    injurymechanism["clotheslining"].replace('Y', 1, inplace=True)
    injurymechanism["clotheslining"].replace('ND', -1, inplace=True)
    injurymechanism["clotheslining"].fillna(-1, inplace=True)
    injurymechanism["helmet"].replace('N', 0, inplace=True)
    injurymechanism["helmet"].replace('Y', 1, inplace=True)
    injurymechanism["helmet"].replace('ND', -1, inplace=True)
    injurymechanism["headfirst"].replace('N', 0, inplace=True)
    injurymechanism["headfirst"].replace('Y', 1, inplace=True)
    injurymechanism["headfirst"].replace('ND', -1, inplace=True)
    injurymechanism["headfirstregion"].replace('ND', -1, inplace=True)
    injurymechanism["headfirstregion"].replace('B', 0, inplace=True)
    injurymechanism["headfirstregion"].replace('F', 1, inplace=True)
    injurymechanism["headfirstregion"].replace('T', 2, inplace=True)
    injurymechanism["headfirstregion"].replace('S', 3, inplace=True)
    injurymechanism.drop(["injurydate", "injurytime"], axis=1, inplace=True)
    return injurymechanism

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
            df[cols] = df[cols].replace('Y', 1)
            df[cols] = df[cols].replace('N', 0)
    return df


def clean_clinical_site(df):
    df = binarize(df)
    
    # Uncomment and fix the GCS mappings
    eye_mapping = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        np.nan: -1
    }
    if 'gcseye' in df.columns:
        df.gcseye = df.gcseye.map(eye_mapping).fillna(-1)
    
    verbal_mapping = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        np.nan: -1
    }
    if 'verbalgcs' in df.columns:
        df.verbalgcs = df.verbalgcs.map(verbal_mapping).fillna(-1)
    
    # Update other mappings to use numeric values
    loc_mapping = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        np.nan: -1
    }
    if 'locevalphysician' in df.columns:
        df.locevalphysician = df.locevalphysician.map(loc_mapping).fillna(-1)

    # ... update other mappings similarly ...
    
    return df

def clean_demo(demo):
    demo_cleaned = binarize(demo)

    return demo_cleaned

def clean_injuryclass(injury_class):
    injury_class_cleaned = binarize(injury_class_cleaned)

    return injury_class_cleaned

def main():
    """
    Main function to execute the cleaning process
    """
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
    df_cleaned = clean_kappa(df_cleaned)
    df_cleaned = clean_medical_history(df_cleaned)
    df_cleaned = clean_injury_mechanism(df_cleaned)

    # Cleaning some parts of totalgcs
    df_cleaned["totalgcs"] = df_cleaned["totalgcs"].replace({"7T":7}).astype(float)

    
    # Verify the cleaning process
    verify_cleaning(df_total, df_cleaned)
    
    os.makedirs("../data", exist_ok=True)
    df_cleaned.to_csv("../data/merged_data_cleaned.csv", index=False)
    print("\nData cleaning completed successfully.")
    print("\nData shape after cleaning:", df_cleaned.shape)

if __name__ == "__main__":
    main()
    