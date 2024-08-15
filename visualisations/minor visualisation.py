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

# load csv functions
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

# directory for folder
target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked'
folder_path = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\data_exploration'
target_file = 'oecd healthcare spending.csv'
path = os.path.join(target_directory, target_file)
os.chdir(target_directory)

# loading some example data
ref = load_csv2('000_reference_data.csv')
age = ref[['age']]

adm = load_csv2('111_admissions.csv')
adm = adm[['HADM_ID','GENDER','INSURANCE','RELIGION','insurance_type','religion_group','age_range']]
pre = load_csv2('316_prescriptions.csv')

# processing the example data
dat = pd.concat([age, adm], axis=1)
dat = dat[['HADM_ID', 'age', 'age_range','GENDER','INSURANCE','RELIGION','insurance_type','religion_group']]

dat2 = pre[['HADM_ID', 'ROUTE', 'time_since_admit', 'seq_num', 'unique_ndc', 'unique_route_per_hadm_id', 'total_ndc',
            'mean_dose_val_rx', 'max_dose_val_rx', 'total_dose_val_rx', 'cumulative_dose_val_rx']]

# printing the example data
print(dat.sample(6))
print(dat2.sample(6))


# finding the columns for the full data
#fulldat = load_csv2('901_cleaned_combined_data72_5_20.csv')
fulldat = load_csv2('902_full_data_pred_72_5.csv')


fullcols = fulldat.columns
lencols = len(fullcols)


numcols = fulldat.select_dtypes(include=[np.number]).columns
len_numcols = len(numcols)



print(fullcols)
print(f'all cols:{lencols}')

print(numcols)
print(f'all cols: {lencols}')
print(f'numeric cols: {len_numcols}')
print(f'cat cols: {lencols-len_numcols}')