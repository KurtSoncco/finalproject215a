import os
import pandas as pd

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

# Cleaning by Kurt

# Analysis variables
def clean_analysis(df_analysis):
    
    # Fill missing values with 0
    df_analysis.fillna(0, inplace=True)

    return df_analysis



# Clinical on Field

def clean_onfield(df_field):
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
    df_outside.loc[df_outside["totalgcsavailable"] =="Y", "totalgcs"] = df_outside[df_outside["totalgcsavailable"] =="Y"]["totalgcsmanual"]
    df_outside.drop(columns=["totalgcsmanual", "sectiongcsavailable", "totalgcsavailable", "gcseye", "verbalgcs", "motorgcs"], inplace=True)

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
    
    # imputing missing values with mode
    for col in clinical_cleaned.select_dtypes(include=['object', 'category']):
        clinical_cleaned[col].fillna(clinical_cleaned[col].mode()[0], inplace=True)

    # imputing missing values with median
    for col in clinical_cleaned.select_dtypes(include=['float64', 'int64']):
        clinical_cleaned[col].fillna(clinical_cleaned[col].median(), inplace=True)

    df = clinical_cleaned.drop(columns=["ArrivalDate","ArrivalTime", "ArrivalTimeND", "ModeArrival"])

    binarize(df)

    # GCSEye
    eye_mapping = {
            1 : np.nan,
            2 : "Pain",
            3 : "Verbal",
            4 : "Spontaneous"
    }
    df.GCSEye = df.GCSEye.map(eye_mapping)
    
    # VerbalGCS
    verbal_mapping = {
            1 : np.nan,
            2 : "Incomprehensible sounds - moans",
            3 : "Inappropriate words - cries to pain",
            4 : "Confused - irritable/cries",
            5 : "Oriented - coos/babbles"
    }
    df.VerbalGCS = df.VerbalGCS.map(verbal_mapping)
    
    # LocEvalPhysician
    loc_mapping = {
            1 : "ED",
            2 : "ICU",
            3 : "General floor",
            4 : "Outpatient Clinic",
            5 : "Other"
    }
    df.LocEvalPhysician = df.LocEvalPhysician.map(loc_mapping) 
    
    # CervicalSpineImmobilization
    immobilization_mapping = {
            1 : 1,
            2 : 1,
            3 : 0
    }
    df.CervicalSpineImmobilization = df.CervicalSpineImmobilization.map(immobilization_mapping)
        
    # IntubatedSS
    intubated_mapping = {
            "Y" : 1,
            "NOTUB" : 0,
            "INTUB" : "Intubation continued",
            "EXTUB" : "Extubated"
    }
    df.IntubatedSS = df.IntubatedSS.map(intubated_mapping)
        
    # CSpinePrecautions
    cspine_mapping = {
            "YD" : 1,
            "N" : 0,
            "YND" : "ND"
    }
    df.CSpinePrecautions = df.CSpinePrecautions.map(cspine_mapping)
    
    # PtSensoryLoss
    sensory_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "ND" : "ND"
    }
    df.PtSensoryLoss = df.PtSensoryLoss.map(sensory_mapping)
        
    # PtParesthesias
    paresthesias_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "ND" : "ND"
    }
    df.PtParesthesias = df.PtParesthesias.map(paresthesias_mapping)
        
    # LimitedRangeMotion
    range_mapping = {
            1 : 1,
            0 : 0,
            "3" : "S",
            "4" : "C-collar in place",
            "NA" : "NA",
            "ND" : "ND"
    }
    df.LimitedRangeMotion = df.LimitedRangeMotion.map(range_mapping)    
    
    # MotorGCS
    motor_mapping = {
            1 : np.nan,
            2 : "Abnormal extension posturing",
            3 : "Abnormal flexure posturing",
            4 : "Withdraws to pain",
            5 : "Localizes pain [withdraws to touch]",
            6 : "Follow Commands"
    } 
    df.MotorGCS = df.MotorGCS.map(motor_mapping)

    for col in df.select_dtypes(include=['object', 'category']):
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col].fillna(df[col].median(), inplace=True)

    return df

def clean_demo(demo):
    # threshold for dropping columns with excessive missing data
    threshold = 80.0
    
    # drop columns w/ missing above threshold
    demo_cleaned = demo.loc[:, (demo.isnull().mean() * 100) <= threshold]

    # imputing missing values with mode
    for col in demo_cleaned.select_dtypes(include=['object', 'category']):
        demo_cleaned[col].fillna(demo_cleaned[col].mode()[0], inplace=True)

    # imputing missing values with median
    for col in demo_cleaned.select_dtypes(include=['float64', 'int64']):
        demo_cleaned[col].fillna(demo_cleaned[col].median(), inplace=True)

    binarize(demo_cleaned)

    return demo_cleaned

def clean_injuryclass(injury_class):
    # threshold for dropping columns with excessive missing data
    threshold = 80.0
    
    # drop columns w/ missing above threshold
    injury_class_cleaned = injury_class.loc[:, (injury_class.isnull().mean() * 100) <= threshold]
    
    # imputing missing values with mode
    for col in injury_class_cleaned.select_dtypes(include=['object', 'category']):
        injury_class_cleaned[col].fillna(injury_class_cleaned[col].mode()[0], inplace=True)

    # imputing missing values with median
    for col in injury_class_cleaned.select_dtypes(include=['float64', 'int64']):
        injury_class_cleaned[col].fillna(injury_class_cleaned[col].median(), inplace=True)

    binarize(injury_class_cleaned)

    return injury_class_cleaned

#


