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
schema = 0

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
if schema == 1:
    # load in the files
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

    # define schema
    schema = {
        "admissions": [col.lower() for col in list(df_h_adm.columns)],
        "callout": [col.lower() for col in list(df_h_cal.columns)],
        "chartevents": [col.lower() for col in list(df_h_cha.columns)],
        "cptevents": [col.lower() for col in list(df_h_cpt.columns)],
        "datetimeevents": [col.lower() for col in list(df_h_dte.columns)],
        "diagnoses_icd": [col.lower() for col in list(df_h_dia.columns)],
        "drgcodes": [col.lower() for col in list(df_h_drg.columns)],
        "icustays": [col.lower() for col in list(df_h_icu.columns)],
        "inputevents_cv": [col.lower() for col in list(df_h_icv.columns)],
        "inputevents_mv": [col.lower() for col in list(df_h_imv.columns)],
        "labevents": [col.lower() for col in list(df_h_lab.columns)],
        "microbiologyevents": [col.lower() for col in list(df_h_mic.columns)],
        "noteevents": [col.lower() for col in list(df_h_not.columns)],
        "outputevents": [col.lower() for col in list(df_h_out.columns)],
        "patients": [col.lower() for col in list(df_h_pat.columns)],
        "prescriptions": [col.lower() for col in list(df_h_pre.columns)],
        "procedureevents_mv": [col.lower() for col in list(df_h_pem.columns)],
        "procedures_icd": [col.lower() for col in list(df_h_pic.columns)],
        "services": [col.lower() for col in list(df_h_ser.columns)],
        "transfers": [col.lower() for col in list(df_h_tra.columns)]
    }

    # function to filter linked columns
    def filter_linked_columns(schema, relationships):
        linked_columns = {}
        for (table, column), ref_table in relationships.items():
            column = column.lower()
            if table in linked_columns:
                linked_columns[table].add(column)
            else:
                linked_columns[table] = set([column])

            if ref_table in linked_columns:
                linked_columns[ref_table].add(column)
            else:
                linked_columns[ref_table] = set([column])

        filtered_schema = {table: [col.lower() for col in schema[table] if col.lower() in linked_columns.get(table, set())]
                           for table in schema}
        return filtered_schema, linked_columns

    # schema column names to lowercase
    normalized_schema = {table: [col.lower() for col in columns] for table, columns in schema.items()}

    # relationships, big list, hopefully got them all
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
        ("callout", "icustay_id"): "icustays"
    }

    # filter schema to linked columns
    filtered_schema, linked_columns = filter_linked_columns(normalized_schema, relationships)


    # arrow colour map
    color_map = {
        'hadm_id': 'red',
        'icustay_id': 'blue',
        'subject_id': 'green',
        'cgid': 'purple',
        'itemid': 'orange',
        'icd9_code': 'brown',
    }

    # create graph and params
    schema_image = graphviz.Digraph('schema', node_attr={'shape': 'plaintext'})
    schema_image.attr(rankdir='TB', ranksep='0.5')
    fixed_width = 130

    # Add tables with linked columns and count of unlinked columns
    for table, columns in normalized_schema.items():
        linked_cols = filtered_schema.get(table, [])
        unlinked_columns_count = len([col for col in columns if col not in linked_columns.get(table, set())])

        if not linked_cols and unlinked_columns_count == 0:
            continue

        label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" WIDTH="{fixed_width}">
                    <TR><TD COLSPAN="2" BGCOLOR="lightblue"><B>{table}</B></TD></TR>'''
        for col in linked_cols:
            label += f'<TR><TD PORT="{col}" BGCOLOR="#e6f7ff" ALIGN="CENTER" WIDTH="{fixed_width}">{col}</TD></TR>'
        if unlinked_columns_count > 0:
            label += f'<TR><TD COLSPAN="2" BGCOLOR="lightgrey" ALIGN="CENTER" WIDTH="{fixed_width}">{unlinked_columns_count} unlinked columns</TD></TR>'
        label += '</TABLE>>'

        print(f'adding node {table} with columns: {linked_cols} and {unlinked_columns_count} unlinked columns')
        schema_image.node(table, label=label)


    # relationships arrows
    for (table, column), ref_table in relationships.items():
        column = column.lower()
        color = color_map.get(column, 'black')  # defaults to black if there's no predetermined colour
        print(f'Connecting {table}:{column} to {ref_table}:{column} with color {color}')
        schema_image.edge(f'{table}:{column}', f'{ref_table}:{column}',
                        color=color, arrowhead='normal', arrowtail='dot')

    # save the graph as a .png file. it saves twice for some reason, no clue why
    output_path = os.path.join(folder_path, 'mimic_full_schema_visual')
    schema_image.render(output_path, format='png', cleanup=False)


df_ref = load_csv2('000_reference_data.csv')



median_los_days = df_ref['los_hours'].median() / 24
mean_los_days = df_ref['los_hours'].mean() / 24
max_los_days = df_ref['los_hours'].max() / 24
max_los_hrs = df_ref['los_hours'].max()
min_los_days = df_ref['los_hours'].min() / 24
min_los_hrs = df_ref['los_hours'].min()
median_los_hrs = df_ref['los_hours'].median()
mean_los_hrs = df_ref['los_hours'].mean()

total_count = len(df_ref)
short_stay = (df_ref['los_hours'] < 168).sum()
medium_stay = ((df_ref['los_hours'] >= 168) & (df_ref['los_hours'] <= 504)).sum()
long_stay = (df_ref['los_hours'] > 504).sum()

percent_short_stay = (short_stay / total_count) * 100
percent_medium_stay = (medium_stay / total_count) * 100
percent_long_stay = (long_stay / total_count) * 100

longer_than_median = (df_ref['los_hours'] > median_los_hrs).sum()
percent_longer_than_median = (longer_than_median / total_count) * 100

print(f'Mean LOS: {mean_los_days} days ({mean_los_hrs} hours)')
print(f'Median LOS: {median_los_days} days ({median_los_hrs} hours)')
print(f'Max LOS: {max_los_days} days ({max_los_hrs} hours)')
print(f'Min LOS: {min_los_days} days ({min_los_hrs} hours)')

print(f'Short stay (< 7 days): {short_stay} ({percent_short_stay}%)')
print(f'Medium stay (7-21 days): {medium_stay} ({percent_medium_stay}%)')
print(f'Long stay (> 21 days): {long_stay} ({percent_long_stay}%)')

print(f'Patients staying longer than median: {longer_than_median} ({percent_longer_than_median}%)')


df_adm = load_csv2('101_admissions.csv')

total_diagnoses = df_adm['DIAGNOSIS'].nunique()

print(f'Total Diagnoses: {total_diagnoses}')