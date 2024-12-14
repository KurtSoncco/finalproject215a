import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set colorblind color palette
sns.set_palette("colorblind")

# Cervical spine injuries 
def plot_csi_site():
    """Plot the number of cervical spine injuries by site."""
    
    # Injury data
    injury_data = pd.read_csv("../data/CSpine/CSV datasets/injuryclassification.csv")

    # Convert Nan values to 0, and Y/N to 1/0
    injury_data = injury_data.fillna(0)
    injury_data = injury_data.replace({'Y': 1, 'N': 0}).infer_objects(copy=False)
    injury_data.columns = injury_data.columns.str.lower()

    # Eliminate any dtype object columns
    injury_data = injury_data.select_dtypes(exclude=['object'])

    # Group the data by site and sum the values
    injury_site = injury_data.groupby(["site"]).sum()[["csfractures","ligamentoptions"]]

    # Melt the data
    injury_site = pd.melt(injury_site.reset_index(), id_vars=["site"], value_vars=["csfractures","ligamentoptions"])

    # Change the variable names
    injury_site = injury_site.replace({"csfractures": "Cervical Spine Fractures", "ligamentoptions": "Ligamentous Injuries"})

    # Plot the data
    custom_palette = ["#ADD8E6", "#FFA07A"]  # light blue and light orange
    sns.barplot(x="site", y="value", hue="variable", data=injury_site, palette=custom_palette)
    plt.title("Cervical Spine Injuries by Site")
    plt.ylabel("Count")
    plt.xlabel("Hospital Site")
    plt.ylim([0, 40])
    plt.legend(title="Injury Type")
    plt.savefig("../plots/cervical_spine_injuries_by_site.pdf", bbox_inches="tight", dpi=300)

def plot_cases_site(data_target):
    """Plot the proportion of cases per hospital site."""
    # Create a groupby object
    cases_per_site = data_target.groupby("site").sum()["csfractures_y"] / data_target.groupby("site").count()["csfractures_y"]

    # Plot the data
    sns.barplot(x=cases_per_site.index, y=cases_per_site.values, color="lightblue")
    plt.xlabel("Hospital Site")
    plt.ylabel("Proportion of Cases")
    plt.title("Proportion of Cases per Hospital Site")
    plt.title("Proportion of Cases per Hospital Site")
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    plt.savefig("../plots/cases_per_site.pdf", bbox_inches="tight", dpi=300)

def plot_cases_gcs(data_target):
    """Plot the number of cases by GCS score."""
    # Create a groupby object
    cases_gcs = data_target.groupby("totalgcs")["csfractures_y"].sum()

    # Eliminate blank values
    cases_gcs = cases_gcs.drop(-1)

    # Plot the data
    sns.barplot(x=cases_gcs.index, y=cases_gcs.values, color="lightgreen")
    plt.xlabel("GCS Score")
    plt.ylabel("Number of Cases")
    plt.title("Number of Cases by GCS Score")
    plt.savefig("../plots/cases_by_gcs.pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    # Read data
    data = pd.read_csv("../data/merged_data_cleaned.csv", low_memory=False)
    # Read target data
    target = pd.read_csv("../data/target.csv")

    # Convert the columns to lower case
    target.columns = target.columns.str.lower()

    # Plot the cervical spine injuries by site
    plot_csi_site()

    # Get distribution of target variable per site in data
    data_target = pd.merge(data, target, on="studysubjectid")
    
    # Plot the proportion of cases per hospital site
    plot_cases_site(data_target)

    # Plot the number of cases by GCS score
    plot_cases_gcs(data_target)

    # Correlation plots



    





