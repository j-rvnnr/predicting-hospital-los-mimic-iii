import gc
import time
import os
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kruskal
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import graphviz
from IPython.display import Image


# setting pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# load csv function
def load_csv(file_name):
    print(f'loading {file_name}')
    return pd.read_csv(file_name, low_memory=False)

def load_csv2(file_name, subdirectory='output_files'):
    file_path = os.path.join(target_directory, subdirectory, file_name)
    print(f'loading {file_path}')
    return pd.read_csv(file_path, low_memory=False)

def load_csv3(file_name):
    print(f'loading header of {file_name}')
    return pd.read_csv(file_name, nrows=0)

target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked'
folder_path = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\data_exploration'

target_file = 'oecd healthcare spending.csv'
path = os.path.join(target_directory, target_file)
os.chdir(target_directory)

# set which misc plots/visualisations we wish to do
oecd = 0
admission_nums = 1
schema = 1

# oecd visualiser
if oecd == 1:
    df = load_csv(path)
    df_cols = df[['Reference area', 'REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]

    # sorting by 'reference area' and 'time_period' in descending order
    df_cols = df_cols.sort_values(by=['Reference area', 'TIME_PERIOD'], ascending=[True, False])

    # list of oecd countries
    oecd_countries = [
        'Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
        'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands',
        'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden',
        'Switzerland', 'Turkey', 'United Kingdom', 'United States'
    ]

    # filtering the dataframe to include only oecd countries
    df_oecd = df_cols[df_cols['Reference area'].isin(oecd_countries)]

    # filter the dataframe to include only the specified countries
    selected_countries = ['United States', 'United Kingdom', 'Canada', 'Germany']
    df_selected = df_oecd[df_oecd['Reference area'].isin(selected_countries)]

    # calculate the oecd average
    df_oecd_avg = df_oecd.groupby('TIME_PERIOD')['OBS_VALUE'].mean().reset_index()
    df_oecd_avg['Reference area'] = 'OECD Average'

    print(df_oecd_avg)

    # combine the selected countries data with the oecd average data
    df_combined = pd.concat([df_selected, df_oecd_avg])

    # plot the data
    plt.figure(figsize=(14, 8))

    # iterate over each country and plot
    for country in selected_countries + ['OECD Average']:
        country_data = df_combined[df_combined['Reference area'] == country]
        plt.plot(country_data['TIME_PERIOD'], country_data['OBS_VALUE'], label=country, marker='.')

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    # labels and titles
    plt.xlabel('Year')
    plt.ylabel('% of GDP')
    plt.title('Healthcare Spending as % of GDP')
    plt.legend()

    # display the plot
    plt.show()

# total admissions and different patients
if admission_nums == 1:
    df_adm_file = load_csv('ADMISSIONS.csv')
    print(f'unique patitnt: {df_adm_file['SUBJECT_ID'].nunique()}')
    print(f'unique hosptial visits: {df_adm_file['HADM_ID'].nunique()}')


# schema

df_h_adm = load_csv3('ADMISSIONS.csv')
df_h_cal = load_csv3('CALLOUT.csv')
df_h_car = load_csv3('CAREGIVERS.csv')
df_h_cha = load_csv3('CHARTEVENTS.csv')
df_h_cpt = load_csv3('CPTEVENTS.csv')
df_h_dte = load_csv3('DATETIMEEVENTS.csv')
df_h_dcp = load_csv3('D_CPT.csv')
df_h_did = load_csv3('D_ICD_DIAGNOSES.csv')
df_h_dip = load_csv3('D_ICD_PROCEDURES.csv')
df_h_dit = load_csv3('D_ITEMS.csv')
df_h_dla = load_csv3('D_LABITEMS.csv')
df_h_dia = load_csv3('DIAGNOSES_ICD.csv')
df_h_drg = load_csv3('DRGCODES.csv')
df_h_icu = load_csv3('ICUSTAYS.csv')
df_h_icv = load_csv3('INPUTEVENTS_CV.csv')
df_h_imv = load_csv3('INPUTEVENTS_MV.csv')
df_h_lab = load_csv3('LABEVENTS.csv')
df_h_mic = load_csv3('MICROBIOLOGYEVENTS.csv')
df_h_not = load_csv3('NOTEEVENTS.csv')
df_h_out = load_csv3('OUTPUTEVENTS.csv')
df_h_pat = load_csv3('PATIENTS.csv')
df_h_pre = load_csv3('PRESCRIPTIONS.csv')
df_h_pem = load_csv3('PROCEDUREEVENTS_MV.csv')
df_h_pic = load_csv3('PROCEDURES_ICD.csv')
df_h_ser = load_csv3('SERVICES.csv')
df_h_tra = load_csv3('TRANSFERS.csv')

schema = {
    "admissions": list(df_h_adm.columns),
    "callout": list(df_h_cal.columns),
    "caregivers": list(df_h_car.columns),
    "chartevents": list(df_h_cha.columns),
    "cptevents": list(df_h_cpt.columns),
    "datetimeevents": list(df_h_dte.columns),
    "d_icd_diagnoses": list(df_h_did.columns),
    "d_icd_procedures": list(df_h_dip.columns),
    "d_items": list(df_h_dit.columns),
    "d_labitems": list(df_h_dla.columns),
    "diagnoses_icd": list(df_h_dia.columns),
    "drgcodes": list(df_h_drg.columns),
    "icustays": list(df_h_icu.columns),
    "inputevents_cv": list(df_h_icv.columns),
    "inputevents_mv": list(df_h_imv.columns),
    "labevents": list(df_h_lab.columns),
    "microbiologyevents": list(df_h_mic.columns),
    "noteevents": list(df_h_not.columns),
    "outputevents": list(df_h_out.columns),
    "patients": list(df_h_pat.columns),
    "prescriptions": list(df_h_pre.columns),
    "procedureevents_mv": list(df_h_pem.columns),
    "procedures_icd": list(df_h_pic.columns),
    "services": list(df_h_ser.columns),
    "transfers": list(df_h_tra.columns)
}

# oh boy this is a big old list. Hope it all works, I think it's correct
relationships = {
    ("admissions", "subject_id"): "patients",
    ("admissions", "hadm_id"): "icustays",
    ("admissions", "hadm_id"): "diagnoses_icd",
    ("admissions", "hadm_id"): "procedures_icd",
    ("admissions", "hadm_id"): "prescriptions",
    ("admissions", "hadm_id"): "labevents",
    ("admissions", "hadm_id"): "microbiologyevents",
    ("admissions", "hadm_id"): "noteevents",
    ("admissions", "hadm_id"): "cptevents",
    ("admissions", "hadm_id"): "procedureevents_mv",
    ("admissions", "hadm_id"): "inputevents_mv",
    ("admissions", "hadm_id"): "outputevents",
    ("admissions", "hadm_id"): "chartevents",
    ("admissions", "hadm_id"): "datetimeevents",
    ("admissions", "hadm_id"): "drgcodes",
    ("admissions", "hadm_id"): "services",
    ("admissions", "hadm_id"): "transfers",
    ("icustays", "subject_id"): "patients",
    ("icustays", "hadm_id"): "admissions",
    ("icustays", "icustay_id"): "chartevents",
    ("icustays", "icustay_id"): "datetimeevents",
    ("icustays", "icustay_id"): "inputevents_mv",
    ("icustays", "icustay_id"): "outputevents",
    ("icustays", "icustay_id"): "prescriptions",
    ("icustays", "icustay_id"): "procedureevents_mv",
    ("icustays", "icustay_id"): "labevents",
    ("icustays", "icustay_id"): "microbiologyevents",
    ("icustays", "icustay_id"): "noteevents",
    ("icustays", "icustay_id"): "transfers",
    ("diagnoses_icd", "subject_id"): "patients",
    ("diagnoses_icd", "hadm_id"): "admissions",
    ("procedures_icd", "subject_id"): "patients",
    ("procedures_icd", "hadm_id"): "admissions",
    ("chartevents", "subject_id"): "patients",
    ("chartevents", "hadm_id"): "admissions",
    ("chartevents", "icustay_id"): "icustays",
    ("labevents", "subject_id"): "patients",
    ("labevents", "hadm_id"): "admissions",
    ("labevents", "icustay_id"): "icustays",
    ("prescriptions", "subject_id"): "patients",
    ("prescriptions", "hadm_id"): "admissions",
    ("prescriptions", "icustay_id"): "icustays",
    ("outputevents", "subject_id"): "patients",
    ("outputevents", "hadm_id"): "admissions",
    ("outputevents", "icustay_id"): "icustays",
    ("microbiologyevents", "subject_id"): "patients",
    ("microbiologyevents", "hadm_id"): "admissions",
    ("microbiologyevents", "icustay_id"): "icustays",
    ("noteevents", "subject_id"): "patients",
    ("noteevents", "hadm_id"): "admissions",
    ("noteevents", "icustay_id"): "icustays",
    ("cptevents", "subject_id"): "patients",
    ("cptevents", "hadm_id"): "admissions",
    ("procedureevents_mv", "subject_id"): "patients",
    ("procedureevents_mv", "hadm_id"): "admissions",
    ("procedureevents_mv", "icustay_id"): "icustays",
    ("inputevents_mv", "subject_id"): "patients",
    ("inputevents_mv", "hadm_id"): "admissions",
    ("inputevents_mv", "icustay_id"): "icustays",
    ("inputevents_cv", "subject_id"): "patients",
    ("inputevents_cv", "hadm_id"): "admissions",
    ("inputevents_cv", "icustay_id"): "icustays",
    ("datetimeevents", "subject_id"): "patients",
    ("datetimeevents", "hadm_id"): "admissions",
    ("datetimeevents", "icustay_id"): "icustays",
    ("drgcodes", "subject_id"): "patients",
    ("drgcodes", "hadm_id"): "admissions",
    ("services", "subject_id"): "patients",
    ("services", "hadm_id"): "admissions",
    ("transfers", "subject_id"): "patients",
    ("transfers", "hadm_id"): "admissions",
    ("transfers", "icustay_id"): "icustays",
    ("callout", "subject_id"): "patients",
    ("callout", "hadm_id"): "admissions",
    ("callout", "icustay_id"): "icustays",
    ("caregivers", "cgid"): "chartevents",
    ("caregivers", "cgid"): "datetimeevents",
    ("caregivers", "cgid"): "inputevents_mv",
    ("caregivers", "cgid"): "outputevents",
    ("caregivers", "cgid"): "procedureevents_mv",
    ("caregivers", "cgid"): "noteevents",
    ("d_icd_diagnoses", "icd9_code"): "diagnoses_icd",
    ("d_icd_procedures", "icd9_code"): "procedures_icd",
    ("d_items", "itemid"): "chartevents",
    ("d_items", "itemid"): "datetimeevents",
    ("d_items", "itemid"): "inputevents_mv",
    ("d_items", "itemid"): "outputevents",
    ("d_items", "itemid"): "procedureevents_mv",
    ("d_items", "itemid"): "noteevents",
    ("d_labitems", "itemid"): "labevents"
}

# create graph
schema_image = graphviz.Digraph()
schema_image.attr(rankdir='LR', ratio='compress', nodesep='0.3', ranksep='0.3')

# add the tables, showing only what we need to see.
for table, columns in schema.items():
    linked_columns = {column for (tbl, column) in relationships.keys() if tbl == table} | \
                     {column for column in relationships.values() if column == table}
    label = f"{table}\n" + "\n".join([col for col in columns if col in linked_columns])
    schema_image.node(table, label=label, shape="box")

# add the relationships
for (table, column), ref_table in relationships.items():
    schema_image.edge(table, ref_table, label=column)

# save the graph as a .png file
output_path = os.path.join(folder_path, 'mimic_full_schema.png')
schema_image.render(output_path, format='png', cleanup=False)
