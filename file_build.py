import gc
import time
import os
import datetime

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kruskal
from sklearn.impute import SimpleImputer

'''
Step 0:
initialisation of variables

start the script timer.
set the pandas display options for console output.
set variables that we use throughout the script.

'''

# setting pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# initialising variables which are used throughout the script.
join_column = 'HADM_ID'

analysis_var = 1  # Set this to 1 to perform the analysis, 0 to skip it
analysis_var2 = 1  # set this to 1 for the second statistical analysis
assessment = 1  # set this to 1 to print off some debug while the files are being generated

# target directory. Set this to wherever the mimic csv files are kept
target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked'

section_name = ''

# ranges for parameters
num_entries = [5]
time_window = [72]
max_cat = [20]
completeness_threshold = 80

exp_exceptions = ['los_hours']

# add the shortstay and longstay cutoffs.
shortstay = 168  # this is 7 days in hours
longstay = 504  # this is 21 days in hours

# set the directory to the working one
os.chdir(target_directory)
print(f'Directory set to: {target_directory}')

'''
Step 1:
Function library.

This section keeps the majority of the functions which are used multiple times throughout the script.
There are some one off scripts present in other locations in the script, but most of the repeat use functions are up
here. 
'''

'''
1.1:
csv management functions
'''


# function for logging output of console.
def log_output(output, log_path):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(output + '\n')
    print(output)


# load csv function
def load_csv(file_name):
    print(f'loading {file_name}')
    return pd.read_csv(os.path.join(target_directory, file_name), low_memory=False)


# load csv function again, but this time the subdirectory is always output_files
def load_csv2(file_name, subdirectory='output_files'):
    file_path = os.path.join(target_directory, subdirectory, file_name)
    print(f'loading {file_path}')
    return pd.read_csv(file_path, low_memory=False)


# save csv function
def save_csv(dataframe, file_name):
    print(f'saving csv: {file_name}')
    return dataframe.to_csv(file_name, index=False)


# function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


'''
1.2
dataframe manipulation functions
'''


# this is a simple merger
def merge_dataframes(core_df, df):
    return pd.merge(core_df, df, on='HADM_ID')


# simple dropper
def drop_column(df, column_name):
    df = df.drop(columns=[column_name])
    return df


# function to drop rows where seq_num is above a given number
def drop_rows_seq_num(df, df_name, num_entries):
    print(f'processing dataframe: {df_name}')

    # sub-function to convert a column to numeric and handle errors
    def convert_to_numeric(series):
        return pd.to_numeric(series, errors='coerce')

    if 'seq_num' in df.columns:
        df['seq_num'] = convert_to_numeric(df['seq_num'])
        df = df[df['seq_num'] <= num_entries]
    elif 'SEQ_NUM' in df.columns:
        df['SEQ_NUM'] = convert_to_numeric(df['SEQ_NUM'])
        df = df[df['SEQ_NUM'] <= num_entries]
    else:
        print('no seq_num found, skipping this operation.')

    print(f'finished processing dataframe: {df_name}')
    return df


# function to drop rows where the time window is above a certain value
def drop_time_window(df, time_window):
    time_since_admit_columns = [col for col in df.columns if 'time_since_admit' in col and 'interaction' not in col]
    for col in time_since_admit_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] <= time_window]
    return df


# removes duplicate columns (not sure if this works)
def remove_duplicate_columns(df):
    df = df.T.drop_duplicates().T
    return df


# merges columns based on the hadm_id column
def append_data_hadm_id(df_main, additional_file, join_column, additional_columns):
    df_additional = pd.read_csv(additional_file, low_memory=False)
    filtered_additional = df_additional[df_additional[join_column].isin(df_main[join_column])]
    columns_to_merge = [join_column] + additional_columns
    filtered_additional = filtered_additional[columns_to_merge]
    merged_df = pd.merge(df_main, filtered_additional, on=join_column, how='left')
    return merged_df


# merges columns based on the subject_id column
def append_data_subject_id(df_main, additional_file, join_column='SUBJECT_ID', additional_columns=None,
                           insert_position=2):
    df_additional = pd.read_csv(additional_file, low_memory=False)
    filtered_additional = df_additional[df_additional[join_column].isin(df_main[join_column])]

    columns_to_merge = [join_column] + additional_columns if additional_columns else [join_column]
    filtered_additional = filtered_additional[columns_to_merge]

    merged_df = pd.merge(df_main, filtered_additional, on=join_column, how='left')
    new_columns = [col for col in filtered_additional.columns if col != join_column]
    cols = list(merged_df.columns)

    for i, col in enumerate(new_columns):
        cols.insert(insert_position + i, cols.pop(cols.index(col)))

    merged_df = merged_df[cols]

    return merged_df


# datetime function
def date_time(df, columns):
    for column in columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df


# function to add a sequence number to every dataframe
def seq_num(df, id_column='HADM_ID', new_column='seq_num'):
    df[new_column] = df.groupby(id_column).cumcount() + 1
    return df


'''
1.3
mapping functions
'''


# religion mapping function
def map_religion(df, column='RELIGION', new_column='religion_group'):
    # pre-mapping nan values to 'not specified'
    df[column] = df[column].fillna('not specified')

    religion_mapping = {
        # christian denominations
        'CATHOLIC': 'Christian',
        'PROTESTANT QUAKER': 'Christian',
        'EPISCOPALIAN': 'Christian',
        'GREEK ORTHODOX': 'Christian',
        'BAPTIST': 'Christian',
        'METHODIST': 'Christian',
        'LUTHERAN': 'Christian',
        'UNITARIAN-UNIVERSALIST': 'Christian',
        '7TH DAY ADVENTIST': 'Christian',
        'ROMANIAN EAST. ORTH': 'Christian',

        # jewish denominations
        'JEWISH': 'Jewish',
        'HEBREW': 'Jewish',

        # other religions
        'BUDDHIST': 'Buddhist',
        'MUSLIM': 'Muslim',
        'HINDU': 'Hindu',

        # specific Christian denominations with unique medical views
        'CHRISTIAN SCIENTIST': 'Christian Scientist',
        'JEHOVAH\'S WITNESS': 'Jehovah\'s Witness',

        # miscellaneous categories
        'NOT SPECIFIED': 'None available',
        'UNOBTAINABLE': 'None available',
        'OTHER': 'Other',
    }

    df[new_column] = df[column].map(religion_mapping).fillna('None available')
    return df


# marital status map function
def map_marital_status(df, column='MARITAL_STATUS', new_column='marital_status_group'):
    # pre-mapping nan values to 'UNKNOWN (DEFAULT)'
    df[column] = df[column].fillna('UNKNOWN (DEFAULT)')

    marital_status_mapping = {
        # married statuses
        'MARRIED': 'Married',
        'LIFE PARTNER': 'Married',

        # dingle statuses
        'SINGLE': 'Single',
        'WIDOWED': 'Single',
        'DIVORCED': 'Single',
        'SEPARATED': 'Single',

        # unknown statuses
        'UNKNOWN (DEFAULT)': 'Unknown',
    }

    df[new_column] = df[column].map(marital_status_mapping).fillna('Single')
    return df


# insurance type mapping function
def map_insurance_type(df, column_name='INSURANCE'):
    public_insurance = ['Medicare', 'Medicaid', 'Government']
    private_insurance = ['Private']
    self_pay_insurance = ['Self Pay']

    def get_insurance_type(insurance):
        if insurance in public_insurance:
            return 'public'
        elif insurance in private_insurance:
            return 'private'
        elif insurance in self_pay_insurance:
            return 'self pay'
        else:
            return 'other'

    df['insurance_type'] = df[column_name].apply(get_insurance_type)
    return df


# item mapping function
def map_items(df):
    # load D_ITEMS.csv from the current working directory
    d_items = pd.read_csv('D_ITEMS.csv')

    # ensure ITEMID is the key for merging
    if 'ITEMID' not in df.columns or 'ITEMID' not in d_items.columns:
        raise ValueError('Both dataframes must contain \'ITEMID\' column')

    merged_df = df.merge(d_items[['ITEMID', 'LABEL']], on='ITEMID', how='left')
    itemid_index = merged_df.columns.get_loc('ITEMID')

    columns = list(merged_df.columns)
    columns.insert(itemid_index + 1, columns.pop(columns.index('LABEL')))
    merged_df = merged_df[columns]

    return merged_df


# map icd9 diagnoses
def map_icd9(df, column_name='ICD9_CODE', new_column_name='icd9_category'):
    # define the ICD-9 mapping ranges and categories. data from wikipedia and I shortened it for ease.
    icd9_mapping = {
        range(1, 140): 'infectious',
        range(140, 240): 'neoplasms',
        range(240, 280): 'endocrine/metabolic',
        range(280, 290): 'blood diseases',
        range(290, 320): 'mental disorders',
        range(320, 390): 'nervous system',
        range(390, 460): 'circulatory system',
        range(460, 520): 'respiratory system',
        range(520, 580): 'digestive system',
        range(580, 630): 'genitourinary system',
        range(630, 680): 'pregnancy complications',
        range(680, 710): 'skin diseases',
        range(710, 740): 'musculoskeletal',
        range(740, 760): 'congenital anomalies',
        range(760, 780): 'perinatal conditions',
        range(780, 800): 'symptoms/signs',
        range(800, 1000): 'injury/poisoning',
        'E': 'external causes',
        'V': 'supplementary factors'
    }

    # function to map ICD-9 code to category
    def get_icd9_category(code):
        if isinstance(code, str):
            if code.startswith('E'):
                return 'external causes'
            elif code.startswith('V'):
                return 'supplementary factors'
            else:
                try:
                    code_int = int(code[:3])  # convert the first 3 characters to an integer (this is the useful data)
                    for icd_range, category in icd9_mapping.items():
                        if isinstance(icd_range, range) and code_int in icd_range:
                            return category
                    return 'unknown'
                except ValueError:
                    return 'unknown'
        else:
            return 'unknown'

    df[new_column_name] = df[column_name].apply(get_icd9_category)

    return df


# mapping icd9 procedures
def map_icd9_procedures(df, column_name='ICD9_CODE', new_column_name='icd9_procedure_category'):
    # define the ICD-9 procedure mapping ranges and categories based on the image
    icd9_procedure_mapping = {
        range(0, 1): 'misc procedures (00)',
        range(1, 6): 'nervous system',
        range(6, 8): 'endocrine system',
        range(8, 17): 'eye operations',
        range(17, 18): 'misc diagnostic/therapeutic',
        range(18, 21): 'ear operations',
        range(21, 30): 'nose/mouth/pharynx',
        range(30, 35): 'respiratory system',
        range(35, 40): 'cardiovascular system',
        range(40, 42): 'hemic/lymphatic system',
        range(42, 55): 'digestive system',
        range(55, 60): 'urinary system',
        range(60, 65): 'male genital organs',
        range(65, 72): 'female genital organs',
        range(72, 76): 'obstetrical procedures',
        range(76, 85): 'musculoskeletal system',
        range(85, 87): 'integumentary system',
        range(87, 100): 'misc diagnostic/therapeutic',
    }

    # function to map ICD-9 code to procedure category
    def get_icd9_procedure_category(code):
        if pd.isna(code):
            return 'unknown'
        try:
            code_str = str(int(code))
            code_int = int(code_str[:2])
            for icd_range, category in icd9_procedure_mapping.items():
                if code_int in icd_range:
                    return category
            return 'unknown'
        except (ValueError, TypeError):
            return 'unknown'

    df[new_column_name] = df[column_name].apply(get_icd9_procedure_category)

    return df


# using a mapping dictionary to make ethnicity a bit more useful
def map_ethnicity(df, column_name='ETHNICITY', new_column_name='ethnicities'):
    # making the ethnicity mapping dictionary
    ethnicity_mapping = {
        'WHITE': 'WHITE',
        'UNKNOWN/NOT SPECIFIED': 'UNKNOWN',
        'MULTI RACE ETHNICITY': 'OTHER',
        'BLACK/AFRICAN AMERICAN': 'BLACK',
        'HISPANIC OR LATINO': 'HISPANIC',
        'PATIENT DECLINED TO ANSWER': 'UNKNOWN',
        'ASIAN': 'ASIAN',
        'OTHER': 'OTHER',
        'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC',
        'ASIAN - VIETNAMESE': 'ASIAN',
        'AMERICAN INDIAN/ALASKA NATIVE': 'NATIVE AMERICAN',
        'WHITE - RUSSIAN': 'WHITE',
        'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
        'ASIAN - CHINESE': 'ASIAN',
        'ASIAN - ASIAN INDIAN': 'ASIAN',
        'BLACK/AFRICAN': 'BLACK',
        'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
        'HISPANIC/LATINO - DOMINICAN': 'HISPANIC',
        'UNABLE TO OBTAIN': 'UNKNOWN',
        'BLACK/CAPE VERDEAN': 'BLACK',
        'BLACK/HAITIAN': 'BLACK',
        'WHITE - OTHER EUROPEAN': 'WHITE',
        'PORTUGUESE': 'WHITE',
        'SOUTH AMERICAN': 'HISPANIC',
        'WHITE - EASTERN EUROPEAN': 'WHITE',
        'CARIBBEAN ISLAND': 'OTHER',
        'ASIAN - FILIPINO': 'ASIAN',
        'ASIAN - CAMBODIAN': 'ASIAN',
        'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 'HISPANIC',
        'WHITE - BRAZILIAN': 'WHITE',
        'ASIAN - KOREAN': 'ASIAN',
        'HISPANIC/LATINO - COLOMBIAN': 'HISPANIC',
        'ASIAN - JAPANESE': 'ASIAN',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'PACIFIC ISLANDER',
        'ASIAN - THAI': 'ASIAN',
        'HISPANIC/LATINO - HONDURAN': 'HISPANIC',
        'HISPANIC/LATINO - CUBAN': 'HISPANIC',
        'MIDDLE EASTERN': 'MIDDLE EASTERN',
        'ASIAN - OTHER': 'ASIAN',
        'HISPANIC/LATINO - MEXICAN': 'HISPANIC',
        'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 'NATIVE AMERICAN'
    }

    # create a new column with the mapped ethnicities
    df[new_column_name] = df[column_name].replace(ethnicity_mapping)

    return df


# map the age to age groups (note that all patients over 89 are classed as 90+ for anonymisation from the data)
def map_age_range(df, age_column='age', new_column_name='age_range'):
    # age range mapping function
    def age_to_range(age):
        if age < 16:
            return None
        elif age <= 18:
            return '16-18'
        elif age <= 29:
            return '19-29'
        elif age <= 39:
            return '30-39'
        elif age <= 49:
            return '40-49'
        elif age <= 59:
            return '50-59'
        elif age <= 69:
            return '60-69'
        elif age <= 79:
            return '70-79'
        elif age <= 89:
            return '80-89'
        else:
            return '90+'

    # apply the age range mapping function
    df[new_column_name] = df[age_column].apply(age_to_range)
    df = df[df[new_column_name].notnull()]

    return df


'''
1.4
feature engineering functions
'''


# calculate los function
def los_func(row, col1, col2):
    if pd.notna(row[col1]) and pd.notna(row[col2]):
        return round((row[col2] - row[col1]).total_seconds() / 3600.0, 3)
    return np.nan


# hospital visit sequence
def make_hospital_visit_seq(df):
    df['hospital_visit'] = df.groupby('SUBJECT_ID').cumcount() + 1
    return df


# max hospital visits
def make_max_hosp_visits(df):
    max_visits = df.groupby('SUBJECT_ID')['hospital_visit'].transform('max')
    df['max_hospital_visits'] = max_visits
    return df


# max sequence number
def make_max_sequence_num(df, df_name):
    seq_num_cols = [col for col in df.columns if col.lower() == 'seq_num']
    if seq_num_cols:
        seq_num_col = seq_num_cols[0]
        df['max_sequence_num'] = df.groupby('HADM_ID')[seq_num_col].transform('max')
        print(f'Added max_sequence_num column to dataframe: {df_name}')
    return df


# unique values per hadm_id
def unique_vals_func(df, id_column='HADM_ID'):
    stats_data = {
        'column': [],
        'max_unique': [],
        'median_unique': [],
        'mean_unique': []
    }

    # group by the specified id column
    grouped = df.groupby(id_column)

    for column in df.columns:
        if column == id_column:
            continue

        # count unique values per id
        unique_counts = grouped[column].nunique()

        # calculate statistics
        max_unique = unique_counts.max()
        median_unique = unique_counts.median()
        mean_unique = unique_counts.mean()

        # store results
        stats_data['column'].append(column)
        stats_data['max_unique'].append(max_unique)
        stats_data['median_unique'].append(median_unique)
        stats_data['mean_unique'].append(mean_unique)

    # convert to dataframe
    stats_df = pd.DataFrame(stats_data)
    print(stats_df)
    return stats_df


# counting unique values
def count_uniques(df):
    unique_counts = {
        'Column': [],
        'Unique Values Count': []
    }

    for column in df.columns:
        unique_counts['Column'].append(column)
        unique_counts['Unique Values Count'].append(df[column].nunique())

    result_df = pd.DataFrame(unique_counts)
    print(result_df)
    return result_df


# this one replaces nans with none, a new variable which can be used in some cases.
def replace_nan_with_none(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna('None Provided')
    return df


# this function makes new total and mean seq num cols
def make_total_mean_seq(df, df_name):
    print(f'Processing dataframe: {df_name}')

    # identify columns that contain 'seq_num' (case insensitive)
    seq_num_cols = [col for col in df.columns if 'seq_num' in col.lower()]

    if seq_num_cols:
        # convert columns to numeric
        for col in seq_num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # calculate the total, mean, and median for each hadm id
        df['total_sequence_num'] = df[seq_num_cols].sum(axis=1)
        df['mean_sequence_num'] = df.groupby('HADM_ID')['total_sequence_num'].transform('mean')
        df['median_sequence_num'] = df.groupby('HADM_ID')['total_sequence_num'].transform('median')

        # calculate the deviation from the mean and median for each column
        for col in seq_num_cols:
            df[f'{col}_dev_mean'] = df[col] - df.groupby('HADM_ID')[col].transform('mean')
            df[f'{col}_dev_median'] = df[col] - df.groupby('HADM_ID')[col].transform('median')

        # calculate the deviation for the total_sequence_num column
        overall_mean_total_seq_num = df['total_sequence_num'].mean()
        overall_median_total_seq_num = df['total_sequence_num'].median()

        df['total_sequence_num_dev_mean'] = df['total_sequence_num'] - overall_mean_total_seq_num
        df['total_sequence_num_dev_median'] = df['total_sequence_num'] - overall_median_total_seq_num
    else:
        print('No seq_num found, skipping this operation.')

    print(f'Finished processing dataframe: {df_name}')
    return df


# this function makes mean and total hours_after cols
def make_total_mean_hours(df, df_name):
    print(f'Processing dataframe: {df_name}')

    # find columns that contain 'hours_after_admit' or 'time_since_admit' (case insensitive)
    hours_cols = [col for col in df.columns if 'hours_after_admit' in col.lower() or 'time_since_admit' in col.lower()]

    if hours_cols:
        for col in hours_cols:
            df[f'total_{col}'] = df[hours_cols].sum(axis=1)
            df[f'mean_{col}'] = df[hours_cols].mean(axis=1)
    else:
        print('No hours_after_admit or time_since_admit found, skipping this operation.')

    print(f'Finished processing dataframe: {df_name}')
    return df


# function to classify stay based on los_hours
def classify_stay(los_hours):
    if los_hours <= shortstay:
        return 0
    elif los_hours <= longstay:
        return 1
    else:
        return 2


# function to calculate the KW and Spearman correlation coefficient stats.
def outlier_stats_output_2(df, target='los_hours', original_df_name='df', exclusions=None, forced_categorical=None,
                           forced_numerical=None):
    if exclusions is None:
        exclusions = []
    if forced_categorical is None:
        forced_categorical = []
    if forced_numerical is None:
        forced_numerical = []

    def convert_to_numeric(df, columns):
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def is_excluded(column):
        return any(substring in column for substring in exclusions)

    factors = [col for col in df.columns if not is_excluded(col)]
    categorical_results = []
    numerical_results = []

    for factor in factors:
        df_non_na = df.drop_duplicates(subset=['HADM_ID']).dropna(subset=[factor])

        if df_non_na.empty:
            continue

        # find if the factor is categorical or numerical
        unique_values = df_non_na[factor].nunique()
        if any(substring in factor for substring in forced_categorical):
            is_categorical = True
        elif any(substring in factor for substring in forced_numerical):
            is_categorical = False
        else:
            is_categorical = df_non_na[factor].dtype == 'object' or unique_values < 0.05 * len(df_non_na)

        if is_categorical:
            # Kruskal-Wallis test for categorical variables
            full_groups = [df_non_na[target][df_non_na[factor] == level] for level in df_non_na[factor].unique()]
            if len(full_groups) > 1:
                full_kruskal_stat, full_kruskal_p_value = kruskal(*full_groups)
            else:
                full_kruskal_stat, full_kruskal_p_value = (None, None)
            full_kruskal_significant = 1 if full_kruskal_p_value is not None and full_kruskal_p_value < 0.05 else 0

            categorical_results.append({
                'factor': factor,
                'length': unique_values,
                'kw_stat': full_kruskal_stat,
                'kw_p_value': full_kruskal_p_value * 100 if full_kruskal_p_value is not None else None,
                # convert to percentage
                'sig_kw': full_kruskal_significant,
                'type': 'categorical',
                'dataframe': original_df_name
            })

        else:
            # Spearman's correlation for numerica variables
            df_non_na = convert_to_numeric(df_non_na, [factor])
            full_spearman_corr, full_spearman_p_value = spearmanr(df_non_na[factor], df_non_na[target])
            full_spearman_significant = 1 if full_spearman_p_value < 0.05 else 0

            numerical_results.append({
                'factor': factor,
                'length': unique_values,
                'spearman_cc': full_spearman_corr,
                'spearman_p_value': full_spearman_p_value * 100,  # convert to percentage
                'sig_spearman': full_spearman_significant,
                'type': 'numerical',
                'dataframe': original_df_name
            })

    categorical_results_df = pd.DataFrame(categorical_results)
    numerical_results_df = pd.DataFrame(numerical_results)

    if categorical_results_df.empty:
        categorical_results_df = pd.DataFrame(
            columns=['factor', 'length', 'kw_stat', 'kw_p_value', 'sig_kw', 'type', 'dataframe'])

    if numerical_results_df.empty:
        numerical_results_df = pd.DataFrame(
            columns=['factor', 'length', 'spearman_cc', 'spearman_p_value', 'sig_spearman', 'type', 'dataframe'])

    return categorical_results_df, numerical_results_df


# function to expand rows to columns
def expand_rows_to_cols(df, df_name, exceptions=None):
    if exceptions is None:
        exceptions = []
    groupby_col = 'HADM_ID'
    columns_to_expand = [col for col in df.columns if col not in exceptions and col not in [groupby_col, 'los_hours']]
    transformed_rows = []

    # extract a clean dataframe name
    clean_df_name = df_name.replace('df_', '').replace('_synth', '')

    print(f'starting to expand rows to columns for {df_name}...')
    total_groups = df[groupby_col].nunique()
    group_counter = 0

    for hadm_id, group in df.groupby(groupby_col):
        row = {groupby_col: hadm_id}
        prev_entry = {}
        for i, (_, entry) in enumerate(group.iterrows()):
            for col in columns_to_expand:
                new_col = f'{clean_df_name}_{col}_{i + 1}'
                if new_col not in row or entry[col] != prev_entry.get(col):
                    row[new_col] = entry[col]
            prev_entry = entry
        transformed_rows.append(row)

        group_counter += 1
        if group_counter % 100 == 0 or group_counter == total_groups:
            print(f'processed {group_counter}/{total_groups} groups for {df_name}')

    transformed_df = pd.DataFrame(transformed_rows)

    max_entries = max(len(group) for _, group in df.groupby(groupby_col))

    for i in range(1, max_entries + 1):
        for col in columns_to_expand:
            col_name = f'{clean_df_name}_{col}_{i}'
            if col_name not in transformed_df.columns:
                transformed_df[col_name] = np.nan

    print(f'finished expanding rows to columns for {df_name}.')
    return transformed_df


# function to filter significant columns
def filter_columns(df, significant_columns):
    columns_to_keep = set(['HADM_ID', 'SUBJECT_ID', 'los_hours']) | set(significant_columns)
    original_columns = set(df.columns)

    # include columns with seq_num in their names
    seq_num_columns = {col for col in df.columns if 'seq_num' in col.lower()}
    # include columns created by other functions
    additional_columns = {col for col in df.columns if 'max_sequence_num' in col or
                          'total_time_since_admit' in col or
                          'mean_time_since_admit' in col or
                          'total_sequence_num' in col or
                          'mean_sequence_num' in col}

    # include categorical columns (assuming categorical columns are strings)
    categorical_columns = {col for col in df.columns if df[col].dtype == 'object'}

    columns_to_keep.update(seq_num_columns)
    columns_to_keep.update(additional_columns)
    columns_to_keep.update(categorical_columns)

    cols_to_retain = [col for col in df.columns if col in columns_to_keep]
    dropped_columns = original_columns - set(cols_to_retain)
    print(f'dropped columns: {dropped_columns}')
    return df[cols_to_retain], dropped_columns


# function to perform stochastic imputation
def stochastic_imputation(df, column):
    observed_values = df[column].dropna()
    if observed_values.empty:
        print(f'no observed values to sample from for column: {column}. skipping imputation for this column.')
        return df[column]  # return the original column without changes
    imputed_values = df[column].apply(lambda x: np.random.choice(observed_values) if pd.isna(x) else x)
    return imputed_values


# makes and names mode columns for categorical features
def make_mode_column(df, column_name):
    # calculate the mode for the specified column, grouped by 'hadm_id'
    mode_series = df.groupby('HADM_ID')[column_name].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'None Provided'
    )

    # merge the calculated mode back into the original DataFrame
    df = df.merge(mode_series, on='HADM_ID', suffixes=('', '_mode')).rename(
        columns={f'{column_name}_mode': f'{column_name.lower()}_mode'}
    )

    return df


# makes binary flags for some features
def make_binary_flag(df, input_col, output_col):
    df[output_col] = df[input_col].apply(lambda x: 1 if x == 1 else 0)


'''
1.5
misc other functions
'''


# delete dataframes for memory
def clear_dataframes(dataframes_to_clear):
    for df_name in dataframes_to_clear:
        if df_name in globals():
            del globals()[df_name]

    gc.collect()
    print('Specified dataframes cleared and memory freed up.')


# completeness function
def completeness_func(df):
    total_rows = len(df)
    completeness_data = {
        'Column': [],
        'Completeness (%)': []
    }

    for column in df.columns:
        non_missing = df[column].notna().sum()
        completeness_data['Column'].append(column)
        completeness_data['Completeness (%)'].append((non_missing / total_rows) * 100)

    complete_df = pd.DataFrame(completeness_data)
    print(complete_df)
    return complete_df


# assessment function
def assess_func(df, log_path):
    # completeness assessment
    complete_df = completeness_func(df)
    log_output('completeness:', log_path)
    log_output(complete_df.to_string(index=False), log_path)

    # unique value assessment
    unique_vals_df = unique_vals_func(df)
    log_output('unique values per HADM_ID:', log_path)
    log_output(unique_vals_df.to_string(index=False), log_path)

    count_uniques_df = count_uniques(df)
    log_output('unique value count:', log_path)
    log_output(count_uniques_df.to_string(index=False), log_path)

    # Print and log sample of the dataframe
    sample_df = df.sample(5)
    log_output('sample data (5):', log_path)
    log_output(sample_df.to_string(index=False), log_path)
    print(sample_df)


'''
step 2:
Processing the data.

This step is where we process the csv files, into a form which we can use later on, for machine learning and further
statistical analysis. It's a three step processing system, where we use the results from the previous step to complete 
the next.

The first step is basic data cleaning, the second step is feature engineering and the final step is synthesising missing
data so our machine learning model has a complete dataset to work with.

The intention is that each step is skipped if the following step's files are already present within the data folder,
however if this part of this comment is present, you can safely assume that I did not have time to implement that, and
as a result, the whole script will be quite slow, as it spends a fair bit of time loading in files which it just
immediately discards. 
'''

# timer to make sure there's no shenanigans afoot
st_time = time.time()
print('time start')

# creating output and log filepaths
out_dir = create_directory('output_files')
log_dir = create_directory(os.path.join(out_dir, 'logs'))

current_date = datetime.datetime.now().strftime('%d-%m-%y')
current_time = datetime.datetime.now().strftime('%H-%M')
log_path = os.path.join(log_dir, f'filebuild_log_{current_date}--{current_time}.txt')

log_output(f'out_dir: {out_dir}', log_path)
log_output(f'log_dir: {log_dir}', log_path)
log_output(f'log_path: {log_path}', log_path)


# first data processing function
# this script is a long one, which basically takes the csv files from the mimic dataset, and transforms them into a form
# which we can use. if these files are already present in the correct folder, then this whole section will be skipped.
def data_process():
    '''
    section 0 - reference stats. These stats are for the data we will use to reference other instances of data.
    reference stats
    '''

    # section title
    section_name = '000_reference_data.csv'

    # loading in the initial csv's and begin to process them
    df_admissions = load_csv('ADMISSIONS.csv')
    df_patients = load_csv('PATIENTS.csv')

    # creating df_ref from selected columns and filtering by HOSPITAL_EXPIRE_FLAG == 0, to ensure living.
    df_ref = df_admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']]
    df_ref = df_ref[df_admissions['HOSPITAL_EXPIRE_FLAG'] == 0]

    # merging df_ref with df_patients to add GENDER and DOB columns
    df_ref = df_ref.merge(df_patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')

    # convert to datetime
    date_time(df_ref, ['ADMITTIME', 'DISCHTIME', 'DOB'])

    # adding los hours and age to the dataframe
    df_ref['los_hours'] = df_ref.apply(lambda row: los_func(row, 'ADMITTIME', 'DISCHTIME'), axis=1)

    # age
    dob = df_ref['DOB'].dt.date
    adt_date = df_ref['ADMITTIME'].dt.date
    df_ref['age'] = (adt_date - dob).apply(lambda x: x.days / 365.25 if pd.notnull(x) else None).round(3)

    # filtering the dataframe so under 16's aren't included, and map using the age mapping function
    df_ref = df_ref[df_ref['age'] >= 16]
    df_ref = map_age_range(df_ref)

    # drop the un-needed columns
    df_ref = df_ref.drop(columns=['DOB', 'DISCHTIME', 'age'])

    # saving the main reference csv. This is filtered data with age > 16 and mapped, and only surviving patients,
    save_csv(df_ref, os.path.join(out_dir, section_name))

    # splitting the data into 'core' and 'outlier' group.
    # finding the median, quartiles and IQR
    median_los = df_ref['los_hours'].median()
    Q1 = df_ref['los_hours'].quantile(0.25)
    Q3 = df_ref['los_hours'].quantile(0.75)
    IQR = Q3 - Q1

    # define the lower and upper bounds for detecting outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # split the dataset into core and outliers based on los_hours
    df_core = df_ref[(df_ref['los_hours'] >= lower_bound) & (df_ref['los_hours'] <= upper_bound)]
    df_outliers = df_ref[(df_ref['los_hours'] < lower_bound) | (df_ref['los_hours'] > upper_bound)]

    # save the core and outlier datasets to CSV files
    save_csv(df_core, os.path.join(out_dir, '001_core_data.csv'))
    save_csv(df_outliers, os.path.join(out_dir, '002_outliers_data.csv'))

    # print summary
    log_output(f'median LOS: {median_los}', log_path)
    log_output(f'full dataset saved with {len(df_ref)} records.', log_path)
    log_output(f'core dataset saved with {len(df_core)} records.', log_path)

    # delete un-needed dataframes, and garbage collect.
    del df_patients
    del df_admissions
    del df_outliers
    gc.collect()

    # set up the main df for the whole file. This should either be df_ref or df_core, unless we're doing something
    # particularly wacky. df_core is default as that's what we're working with for the model.
    df_main = df_ref[['HADM_ID', 'los_hours']]

    '''
    section 1:
    Constructing the patient stays tables. These are as follows:
    Admissions - 101
    Callout - 102
    Icu Stays - 103
    Services - 104
    Transfers - 105
    '''

    # admissions
    section_name = f'101_admissions.csv'

    # if the file exists in the folder, just load it instead of creating a new file
    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:

        additional_file = 'ADMISSIONS.csv'
        additional_columns = ['DISCHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
                              'MARITAL_STATUS', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS']
        df_admissions = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # converting the applicable columns to datetime
        date_time(df_admissions, ['EDREGTIME', 'EDOUTTIME', 'DISCHTIME'])

        # applying the los_func to calculate ed_los
        df_admissions['ed_los'] = df_admissions.apply(lambda row: los_func(row, 'EDREGTIME', 'EDOUTTIME'), axis=1)

        # assessment
        if assessment == 1:
            assess_func(df_admissions, log_path)

        # save the new admissions dataframe
        save_csv(df_admissions, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_admissions
        gc.collect()

    # callout
    section_name = f'102_callout.csv'

    # if the file exists in the folder, just load it instead of creating a new file
    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'CALLOUT.csv'
        additional_columns = ['SUBMIT_CAREUNIT', 'CURR_CAREUNIT', 'CALLOUT_WARDID', 'CALLOUT_SERVICE', 'REQUEST_TELE',
                              'REQUEST_RESP', 'REQUEST_CDIFF', 'REQUEST_MRSA', 'REQUEST_VRE', 'CALLOUT_OUTCOME',
                              'DISCHARGE_WARDID', 'CREATETIME', 'OUTCOMETIME']
        df_callout = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # convert relevant columns to datetime
        date_time(df_callout, ['CREATETIME', 'OUTCOMETIME'])
        df_callout = seq_num(df_callout)

        # assessment
        if assessment == 1:
            assess_func(df_callout, log_path)

        # save the new callout dataframe
        save_csv(df_callout, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_callout
        gc.collect()

    # icu stays
    section_name = f'103_icu_stays.csv'

    # if the file exists in the folder, just load it instead of creating a new file
    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'ICUSTAYS.csv'
        additional_columns = ['ICUSTAY_ID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',
                              'INTIME', 'OUTTIME']
        df_icu = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # convert relevant columns to datetime
        date_time(df_icu, ['INTIME', 'OUTTIME'])
        df_icu = seq_num(df_icu)

        # create icu los using the handy dandy function
        df_icu['icu_los'] = df_icu.apply(lambda row: los_func(row, 'INTIME', 'OUTTIME'), axis=1)

        # assessment
        completeness_func(df_icu)
        unique_vals_func(df_icu)

        # save the new callout dataframe
        save_csv(df_icu, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_icu
        gc.collect()

    # services
    section_name = f'104_services.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

        # if the file does not exist, then we create it.
    else:
        additional_file = 'SERVICES.csv'
        additional_columns = ['PREV_SERVICE', 'CURR_SERVICE', 'TRANSFERTIME']
        df_services = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # date time
        df_services['TRANSFERTIME'] = pd.to_datetime(df_services['TRANSFERTIME'])
        df_services = seq_num(df_services)

        if assessment == 1:
            assess_func(df_services, log_path)

        # save the services dataframe
        save_csv(df_services, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_services
        gc.collect()

    # transfers
    section_name = f'105_transfers.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'TRANSFERS.csv'
        additional_columns = ['ICUSTAY_ID', 'EVENTTYPE', 'PREV_CAREUNIT', 'CURR_CAREUNIT', 'PREV_WARDID', 'CURR_WARDID',
                              'INTIME', 'OUTTIME']
        df_transfers = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # date times
        date_time(df_transfers, ['INTIME', 'OUTTIME'])
        df_transfers = seq_num(df_transfers)

        if assessment == 1:
            assess_func(df_transfers, log_path)

        # save the transfers
        save_csv(df_transfers, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_transfers
        gc.collect()

    '''
    Section 2:
    This is the Critical Care unit Chapter. Although Chartevents is contained within this chapter technically, 
    it probably won't be involved in the processing of this project. The file itself is massive and cumbersome, and more
    trouble than it is worth for modification and editing.

    for this block, we're adding a few features such as time_since_admit. With the full data this can be used as a proxy
    for los hours, but when we're restricting the data by time, this can be helpful. The standard here, is to save
    my own columns in lower case, so I can tell when I've made them and when they're in the dataset by default. This way
    I can keep track of my own information. time_since_admit is calculated prefereably using the charttime, then the 
    starttime, and finally the storetime. 

    Datetimeevents - 1
    input events cv - 2 (iecv)
    input events mv - 3 (iemv)
    noteevents - 4
    output events - 5
    procedure events mv - 6
    '''

    # adding admittime to the reference for the following 2 sections
    df_main = df_main.merge(df_core[['HADM_ID', 'ADMITTIME']], on='HADM_ID', how='left')

    # transfers
    section_name = f'201_datetime.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'DATETIMEEVENTS.csv'
        additional_columns = ['ICUSTAY_ID', 'ITEMID', 'VALUE', 'CHARTTIME', 'STORETIME', 'CGID']
        df_datetime = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_datetime, ['CHARTTIME', 'STORETIME'])
        df_datetime['time_since_admit'] = round((df_datetime['CHARTTIME'] - df_datetime['ADMITTIME']).dt.total_seconds()
                                                / 3600, 3)
        df_datetime = map_items(df_datetime)
        df_datetime = seq_num(df_datetime)

        if assessment == 1:
            assess_func(df_datetime, log_path)

        # save
        df_datetime = df_datetime.drop(columns=['ADMITTIME'])
        save_csv(df_datetime, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_datetime
        gc.collect()

    # input events cv
    section_name = f'202_iecv.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'INPUTEVENTS_CV.csv'
        additional_columns = ['ITEMID', 'AMOUNT', 'RATE', 'CGID', 'ORIGINALAMOUNT', 'ORIGINALROUTE', 'ORIGINALRATE',
                              'ORIGINALSITE', 'STORETIME']
        df_iecv = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_iecv, ['STORETIME'])
        df_iecv = map_items(df_iecv)
        df_iecv = seq_num(df_iecv)

        df_iecv['time_since_admit'] = round((df_iecv['STORETIME'] - df_iecv['ADMITTIME']).dt.total_seconds() / 3600, 3)

        if assessment == 1:
            assess_func(df_iecv, log_path)

        # save
        df_iecv = df_iecv.drop(columns=['ADMITTIME'])
        save_csv(df_iecv, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_iecv
        gc.collect()

    # input events mv
    section_name = f'203_iemv.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'INPUTEVENTS_MV.csv'
        additional_columns = ['STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'RATE', 'STORETIME', 'CGID',
                              'ORDERCATEGORYNAME',
                              'ORDERCOMPONENTTYPEDESCRIPTION', 'ORDERCATEGORYDESCRIPTION', 'TOTALAMOUNT',
                              'PATIENTWEIGHT']
        df_iemv = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_iemv, ['STARTTIME', 'ENDTIME', 'STORETIME'])
        df_iemv['run_time'] = round((df_iemv['ENDTIME'] - df_iemv['STARTTIME']).dt.total_seconds() / 3600, 3)

        df_iemv['time_since_admit'] = round((df_iemv['STARTTIME'] - df_iemv['ADMITTIME']).dt.total_seconds() / 3600, 3)

        df_iemv = map_items(df_iemv)
        df_iemv = seq_num(df_iemv)

        # assessment function
        if assessment == 1:
            assess_func(df_iemv, log_path)

        # save
        df_iemv = df_iemv.drop(columns=['ADMITTIME'])
        save_csv(df_iemv, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_iemv
        gc.collect()

    # input events mv
    section_name = f'204_note_events.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'NOTEEVENTS.csv'
        additional_columns = ['CHARTTIME', 'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID']
        df_notes = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_notes, ['CHARTTIME', 'STORETIME'])
        df_notes['time_since_admit'] = round((df_notes['CHARTTIME'] - df_notes['ADMITTIME']).dt.total_seconds()
                                             / 3600, 3)

        df_notes = seq_num(df_notes)

        # assessment functions
        if assessment == 1:
            assess_func(df_notes, log_path)

        # save
        df_notes = df_notes.drop(columns=['ADMITTIME'])
        save_csv(df_notes, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_notes
        gc.collect()

    # output events
    section_name = f'205_output_events.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'OUTPUTEVENTS.csv'
        additional_columns = ['CHARTTIME', 'ITEMID', 'VALUE', 'STORETIME', 'CGID']
        df_output = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_output, ['CHARTTIME', 'STORETIME'])
        df_output = map_items(df_output)

        df_output['time_since_admit'] = round((df_output['CHARTTIME'] - df_output['ADMITTIME']).dt.total_seconds()
                                              / 3600, 3)
        df_output = seq_num(df_output)

        # assessment functions
        if assessment == 1:
            assess_func(df_output, log_path)

        # save
        df_output = df_output.drop(columns=['ADMITTIME'])
        save_csv(df_output, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_output
        gc.collect()

    # procedure events mv
    section_name = f'206_pemv.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'PROCEDUREEVENTS_MV.csv'
        additional_columns = ['STARTTIME', 'ENDTIME', 'ITEMID', 'VALUE', 'LOCATION', 'LOCATIONCATEGORY', 'CGID',
                              'ORDERCATEGORYNAME', 'ORDERCATEGORYDESCRIPTION']
        df_pemv = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # resolve date time and calculate time since admit
        date_time(df_pemv, ['STARTTIME', 'ENDTIME'])
        df_pemv['time_since_admit'] = round((df_pemv['STARTTIME'] - df_pemv['ADMITTIME']).dt.total_seconds() / 3600, 3)

        # map the itemid's
        df_pemv = map_items(df_pemv)
        df_pemv = seq_num(df_pemv)

        # assessment functions
        if assessment == 1:
            assess_func(df_pemv, log_path)

        # save
        df_pemv = df_pemv.drop(columns=['ADMITTIME'])
        save_csv(df_pemv, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_pemv
        gc.collect()

    '''
    section 3:
    This chapter is where we assemble the Hospital record system tables. When running from fresh, this section is slow,
    as many of these tables are very large. However, we do need the data. 
    cpt events - 1
    icd diagnoses - 2
    drg codes - 3
    labevents - 4
    microbiology events - 5
    prescriptions - 6
    icd procedures - 7
    '''

    # cpt events
    section_name = f'301_cpt_events.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'CPTEVENTS.csv'
        additional_columns = ['COSTCENTER', 'CHARTDATE', 'CPT_CD', 'CPT_NUMBER', 'SECTIONHEADER', 'SUBSECTIONHEADER']
        df_cpt = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # this is a bit ugly and maybe no use since the chartdate is a date.
        df_cpt['CHARTDATE'] = pd.to_datetime(df_cpt['CHARTDATE'], errors='coerce')
        df_cpt['time_since_admit'] = round((df_cpt['CHARTDATE'] - df_cpt['ADMITTIME']).dt.total_seconds() / 3600, 3)

        df_cpt = seq_num(df_cpt)

        # assessment functions
        if assessment == 1:
            assess_func(df_cpt, log_path)

        # save
        df_cpt = df_cpt.drop(columns=['ADMITTIME'])
        save_csv(df_cpt, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_cpt
        gc.collect()

    # icd9 diagnoses. This uses a function from
    section_name = f'302_icd9d.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'DIAGNOSES_ICD.csv'
        additional_columns = ['SEQ_NUM', 'ICD9_CODE']
        df_icd9d = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        map_icd9(df_icd9d, 'ICD9_CODE', 'diagnosis_category')

        # assessment functions
        if assessment == 1:
            assess_func(df_icd9d, log_path)

        # save
        df_icd9d = df_icd9d.drop(columns=['ADMITTIME'])
        save_csv(df_icd9d, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_icd9d
        gc.collect()

    # drg codes
    section_name = f'303_drg.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'DRGCODES.csv'
        additional_columns = ['DRG_TYPE', 'DRG_CODE', 'DESCRIPTION', 'DRG_SEVERITY', 'DRG_MORTALITY']
        df_drg = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        # adding our own seq num, so we have the maximum and total for each hadmid
        df_drg['seq_num'] = df_drg.groupby('HADM_ID').cumcount() + 1

        # assessment functions
        if assessment == 1:
            assess_func(df_drg, log_path)

        # save
        df_drg = df_drg.drop(columns=['ADMITTIME'])
        save_csv(df_drg, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_drg
        gc.collect()

    # lab events
    section_name = f'304_labs.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'LABEVENTS.csv'
        additional_columns = ['ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'FLAG']
        df_labs = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_labs, ['CHARTTIME'])
        df_labs['time_since_admit'] = round((df_labs['CHARTTIME'] - df_labs['ADMITTIME']).dt.total_seconds() / 3600, 3)

        df_labs = seq_num(df_labs)
        df_labs = map_items(df_labs)

        # assessment functions
        assess_func(df_labs, log_path)

        # save
        df_labs = df_labs.drop(columns=['ADMITTIME'])
        save_csv(df_labs, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_labs
        gc.collect()

    # micro events
    section_name = f'305_microbiology.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'MICROBIOLOGYEVENTS.csv'
        additional_columns = ['CHARTDATE', 'CHARTTIME', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ISOLATE_NUM',
                              'AB_ITEMID', 'DILUTION_COMPARISON', 'DILUTION_VALUE', 'INTERPRETATION']
        df_micro = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_micro, ['CHARTTIME', 'CHARTDATE'])

        df_micro['CHARTTIME'] = df_micro['CHARTTIME'].fillna(df_micro['CHARTDATE'])
        df_micro['time_since_admit'] = round((df_micro['CHARTTIME'] - df_micro['ADMITTIME']).dt.total_seconds() / 3600,
                                             3)

        df_micro = seq_num(df_micro)

        # assessment functions
        assess_func(df_micro, log_path)

        # save
        df_micro = df_micro.drop(columns=['ADMITTIME'])
        save_csv(df_micro, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_micro
        gc.collect()

    # prescriptions
    section_name = f'306_prescriptions.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'PRESCRIPTIONS.csv'
        additional_columns = ['STARTDATE', 'DRUG_TYPE', 'DRUG', 'NDC', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'ROUTE']
        df_prescription = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        date_time(df_prescription, ['STARTDATE'])
        df_prescription['time_since_admit'] = round((df_prescription['STARTDATE'] - df_prescription['ADMITTIME']).
                                                    dt.total_seconds() / 3600, 3)
        df_prescription = seq_num(df_prescription)

        # assessment functions
        assess_func(df_prescription, log_path)

        # save
        df_prescription = df_prescription.drop(columns=['ADMITTIME'])
        save_csv(df_prescription, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_prescription
        gc.collect()

    # procedures
    section_name = f'307_procedures.csv'

    if os.path.isfile(os.path.join(out_dir, section_name)):
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)

    # if the file does not exist, then we create it.
    else:
        additional_file = 'PROCEDURES_ICD.csv'
        additional_columns = ['SEQ_NUM', 'ICD9_CODE']
        df_procedures = append_data_hadm_id(df_main, additional_file, join_column, additional_columns)

        map_icd9_procedures(df_procedures)

        # assessment functions
        assess_func(df_procedures, log_path)

        # save
        df_procedures = df_procedures.drop(columns=['ADMITTIME'])
        save_csv(df_procedures, os.path.join(out_dir, section_name))
        time_elapsed = round(time.time() - st_time, 3)
        log_output(f'{section_name} saved at {time_elapsed}', log_path)
        del df_procedures
        gc.collect()


# run the first script
data_process()

# assigning each dataFrame to a variable for direct access
df_core = load_csv2('001_core_data.csv')
df_admissions = load_csv2('101_admissions.csv')
df_callout = load_csv2('102_callout.csv')
df_icustays = load_csv2('103_icu_stays.csv')
df_services = load_csv2('104_services.csv')
df_transfers = load_csv2('105_transfers.csv')
df_datetime = load_csv2('201_datetime.csv')
df_iecv = load_csv2('202_iecv.csv')
df_iemv = load_csv2('203_iemv.csv')
df_note = load_csv2('204_note_events.csv')
df_output = load_csv2('205_output_events.csv')
df_pemv = load_csv2('206_pemv.csv')
df_cpt = load_csv2('301_cpt_events.csv')
df_icd9d = load_csv2('302_icd9d.csv')
df_drg = load_csv2('303_drg.csv')
df_labs = load_csv2('304_labs.csv')
df_microbiology = load_csv2('305_microbiology.csv')
df_prescriptions = load_csv2('306_prescriptions.csv')
df_procedures = load_csv2('307_procedures.csv')

# list of dataframes
dataframes = [
    ('df_admissions', df_admissions),
    ('df_callout', df_callout),
    ('df_icustays', df_icustays),
    ('df_services', df_services),
    ('df_transfers', df_transfers),
    ('df_datetime', df_datetime),
    ('df_iecv', df_iecv),
    ('df_iemv', df_iemv),
    ('df_note', df_note),
    ('df_output', df_output),
    ('df_pemv', df_pemv),
    ('df_cpt', df_cpt),
    ('df_icd9d', df_icd9d),
    ('df_drg', df_drg),
    ('df_labs', df_labs),
    ('df_microbiology', df_microbiology),
    ('df_prescriptions', df_prescriptions),
    ('df_procedures', df_procedures)
]

'''
Step 2.5:
First Analysis section: 

This section only runs if analysis var is set to 1 at the start, otherwise it's skipped.

It runs a kruskal wallis and spearman correlation for each variable and prints them off, allowing me to determine which
features I am going to use for the model analysis. However, it takes a long time to run, so whilst I am doing many
repeats of this file, I will set it to 0 for the time being.
'''

if analysis_var == 1:
    outlier_categorical_results_all = []
    outlier_numerical_results_all = []

    # setting variables for statistical analysis
    target = 'los_hours'

    # these are settings for the statistical and correlation analysis
    forced_categorical = ['ICD9_CODE', 'DRG_CODE', 'NDC', 'CGID', 'PREV_WARDID']

    forced_numerical = ['max_hospital_stay', 'TOTALAMOUNT', 'hospital_visit'
                                                            'AMOUNT', 'VALUENUM', 'DOSE_VAL_RX', 'seq_num', 'VALUE',
                        'DOSE_VAL_RX', 'AMOUNT',
                        'ORIGINALAMOUNT', 'cumulative_timedelta', 'total_ndc', 'unique_ndc', 'unique_route_per_hadm_id',
                        'unique_org_itemid', 'unique_spec_type_desc', 'unique_isolate_num', 'unique_ab_itemid',
                        'unique_spec_type_desc', 'unique_isolate_num', 'unique_org_itemid', 'unique_ab_itemid',
                        'unique_itemid_count', 'unique_value_count', 'unique_itemid_count', 'unique_value_count',
                        'unique_flag_count', 'mean_drg_severity', 'sum_drg_severity', 'max_drg_severity',
                        'mean_drg_mortality', 'ed_los', 'sum_drg_mortality', 'max_drg_mortality',
                        'cumulative_drg_severity', 'cumulative_drg_mortality', 'drg_type_count', 'drg_code_count',
                        'unique_cpt_count', 'unique_subsectionheader_count', 'sum_value_per_hadm_id',
                        'average_value_per_hadm_id', 'rate_to_amount_ratio', 'max_rate_per_hadm_id',
                        'sum_amount_per_hadm_id', 'average_amount_per_hadm_id', 'cgid_count_per_hadm_id',
                        'itemid_count_per_hadm_id', 'total_rate_per_hadm_id', 'total_amount_per_hadm_id',
                        'cumulative_timedelta', 'distinct_careunits_count', 'total_transfers', 'num_transfers']

    exclusions = ['HADM_ID', 'SUBJECT_ID', 'los_hours', 'ADMITTIME', 'STARTDATE', 'ENDDATE', 'STORETIME',
                  'VALUEUOM', 'OUTCOMETIME', 'ICUSTAY_ID', 'TRANSFERTIME', 'STARTTIME', 'ENDTIME',
                  'EDREGTIME', 'EDOUTTIME', 'CHARTTIME', 'CHARTDATE', 'INTIME', 'OUTTIME', 'CREATETIME', 'DISCHTIME',
                  'seq_num', 'SEQ_NUM']

    for df_name, df in dataframes:
        print(f'processing outlier stats for dataframe: {df_name}')
        categorical_results_df_outlier, numerical_results_df_outlier = outlier_stats_output_2(
            df, target=target, original_df_name=df_name, exclusions=exclusions, forced_categorical=forced_categorical,
            forced_numerical=forced_numerical)
        outlier_categorical_results_all.append(categorical_results_df_outlier)
        outlier_numerical_results_all.append(numerical_results_df_outlier)

    # concatenate outlier results
    outlier_categorical_results_df = pd.concat(outlier_categorical_results_all, ignore_index=True).sort_values(
        by='kw_stat', ascending=False)
    outlier_numerical_results_df = pd.concat(outlier_numerical_results_all, ignore_index=True).sort_values(
        by='spearman_cc', ascending=False)

    cat_df_sig = outlier_categorical_results_df[outlier_categorical_results_df['sig_kw'] == 1]
    num_df_sig = outlier_numerical_results_df[outlier_numerical_results_df['sig_spearman'] == 1]

    # save the dataframes as csv files
    out_dir = 'output_files'  # specify the output directory

    # ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save the dataframes
    cat_df_sig.to_csv(os.path.join(out_dir, '003_categorical_significant.csv'), index=False)
    num_df_sig.to_csv(os.path.join(out_dir, '004_numerical_significant.csv'), index=False)

    outlier_categorical_results_df.to_csv(os.path.join(out_dir, '005_categorical_correlation.csv'), index=False)
    outlier_numerical_results_df.to_csv(os.path.join(out_dir, '006_numerical_correlation.csv'), index=False)

    # print results
    print('\nOutlier Categorical Results:')
    print(cat_df_sig)

    print('\nOutlier Numerical Results:')
    print(num_df_sig)

    # group by 'dataframe' column
    grouped_cat_df_sig = cat_df_sig.groupby('dataframe')
    grouped_num_df_sig = num_df_sig.groupby('dataframe')

    # print results
    print('\nOutlier Categorical Results grouped by dataframe:')
    for name, group in grouped_cat_df_sig:
        print(f'\Dataframe: {name}')
        print(group)

    print('\nOutlier Numerical Results grouped by dataframe:')
    for name, group in grouped_num_df_sig:
        print(f'\nDataframe: {name}')
        print(group)

else:
    print('Analysis skipped as analysis_var is set to 0.')

'''
Step 3:
feature engineering section:

This is the main feature engineering section of the script. There have been some light touches above, as things like
mapping/reducing dimensionality could be considered feature engineering, but this is the main section. 

The goal here is to reduce the ratio of categorical to numerical features within the data. As it stands, the final data
is about 75%~ categorical, which when we one-hot encode, becomes a very large very sparse array. This impacts training
time and makes the neural network quite prone to over-fitting. Although in this section, we do have some number of cat
variables being created, the main focus was to increase the number/usefulness of numerical variables. 

At the end, there is another KW and SCC variablility analysis. This is to inspect whether this section has actually done
anything, or if the variables are simply not worth using. 

Final note: all the new columns that I will be creating are in lowercase, as the columns present in the original data
are all uppercase. This makes it easy to quickly visually distinguish which columns are feature engineered, and which
columns are present from the original data.
'''

# load in reference data
core = load_csv(os.path.join(out_dir, '000_reference_data.csv'))
core = drop_column(core, 'los_hours')

# admissions feature engineering
section_name = '111_admissions.csv'

# check if the file exists in the folder
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_admissions = load_csv(os.path.join(out_dir, section_name))
    print(df_admissions.head(5))
else:
    # merging core with admissions specifically to calculate the hospital stays variables.
    df_admissions = merge_dataframes(core, df_admissions)

    # df_admissions feature engineering
    df_admissions = map_religion(df_admissions)
    df_admissions = map_marital_status(df_admissions)
    df_admissions = map_insurance_type(df_admissions)
    df_admissions = replace_nan_with_none(df_admissions, ['LANGUAGE'])
    df_admissions = make_hospital_visit_seq(df_admissions)
    df_admissions = make_max_hosp_visits(df_admissions)

    # save
    save_csv(df_admissions, os.path.join(out_dir, section_name))
    print(df_admissions.head(5))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# feature engineering the callout table
section_name = '112_callout.csv'

# check if the file exists in the folder
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_callout = load_csv(os.path.join(out_dir, section_name))
else:
    # df_callout feature engineering

    df_callout['CREATETIME'] = pd.to_datetime(df_callout['CREATETIME'])
    df_callout['OUTCOMETIME'] = pd.to_datetime(df_callout['OUTCOMETIME'])

    df_callout['timedelta'] = (df_callout['OUTCOMETIME'] - df_callout['CREATETIME']).dt.total_seconds() / 3600

    # interactions
    df_callout['submit_curr_careunit_interaction'] = df_callout['SUBMIT_CAREUNIT'] + '_' + df_callout['CURR_CAREUNIT']
    df_callout['submit_callout_service_interaction'] = df_callout['SUBMIT_CAREUNIT'] + '_' + df_callout[
        'CALLOUT_SERVICE']

    # create flag features
    make_binary_flag(df_callout, 'REQUEST_TELE', 'tele_flag')
    make_binary_flag(df_callout, 'REQUEST_RESP', 'resp_flag')
    make_binary_flag(df_callout, 'REQUEST_CDIFF', 'cdiff_flag')
    make_binary_flag(df_callout, 'REQUEST_MRSA', 'mrsa_flag')
    make_binary_flag(df_callout, 'REQUEST_VRE', 'vre_flag')

    df_callout = replace_nan_with_none(df_callout, ['SUBMIT_CAREUNIT', 'CURR_CAREUNIT', 'CALLOUT_WARDID',
                                                    'CALLOUT_SERVICE', 'CALLOUT_OUTCOME', 'DISCHARGE_WARDID'])

    # cumulative and mean timedelta
    df_callout['cumulative_timedelta'] = df_callout['timedelta'].cumsum()
    df_callout['mean_timedelta'] = df_callout['timedelta'].expanding().mean()

    # save
    print(df_callout.head())
    save_csv(df_callout, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# feature engineering the icy stays
section_name = '113_icustays.csv'

# check if the file exists in the folder
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_icustays = load_csv(os.path.join(out_dir, section_name))
else:
    # feature engineering for icu stays
    # make sure intime and outtime are times. I've had funny business with this in the past.
    df_icustays['INTIME'] = pd.to_datetime(df_icustays['INTIME'])
    df_icustays['OUTTIME'] = pd.to_datetime(df_icustays['OUTTIME'])

    # days and hours, some small effect
    df_icustays['admission_hour'] = df_icustays['INTIME'].dt.hour
    df_icustays['discharge_hour'] = df_icustays['OUTTIME'].dt.hour
    df_icustays['admission_dayofweek'] = df_icustays['INTIME'].dt.dayofweek
    df_icustays['discharge_dayofweek'] = df_icustays['OUTTIME'].dt.dayofweek

    # care unit transitions and interactions with first and last careunit
    df_icustays['careunit_transition'] = (df_icustays['FIRST_CAREUNIT'] != df_icustays['LAST_CAREUNIT']).astype(int)
    df_icustays['careunit_interaction'] = df_icustays['FIRST_CAREUNIT'] + '_' + df_icustays['LAST_CAREUNIT']

    # average and total icu los
    df_icustays['average_icu_los'] = df_icustays.groupby('HADM_ID')['icu_los'].transform('mean')
    df_icustays['average_icu_los'] = df_icustays.groupby('HADM_ID')['icu_los'].transform('median')
    df_icustays['total_icu_los'] = df_icustays.groupby('HADM_ID')['icu_los'].transform('sum')

    # save
    print(df_icustays.head())
    save_csv(df_icustays, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# services feature engineering
section_name = '114_services.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_services = load_csv(os.path.join(out_dir, section_name))
else:
    df_services = replace_nan_with_none(df_services, ['PREV_SERVICE'])

    # create interaction between prev service and curr service
    df_services['service_interaction'] = df_services['PREV_SERVICE'] + '_' + df_services['CURR_SERVICE']

    # number of transfers for each hadm_id
    df_services['num_transfers'] = df_services.groupby('HADM_ID')['HADM_ID'].transform('count')

    df_services = make_mode_column(df_services, 'PREV_SERVICE')
    df_services = make_mode_column(df_services, 'CURR_SERVICE')
    df_services = make_mode_column(df_services, 'service_interaction')

    # save
    print(df_services.head())
    save_csv(df_services, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# transfers feature engineering
section_name = '115_transfers.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_transfers = load_csv(os.path.join(out_dir, section_name))
else:

    # function to handle missing careunit and wardid based on eventtype
    def handle_missing_careunit_wardid(df_transfers):
        def fill_values(row, column_name, event_type, fill_value):
            if pd.isna(row[column_name]) and row['EVENTTYPE'] == event_type:
                return fill_value
            elif pd.isna(row[column_name]):
                return 'not provided'
            else:
                return row[column_name]

        df_transfers['PREV_CAREUNIT'] = df_transfers.apply(
            lambda row: fill_values(row, 'PREV_CAREUNIT', 'admit', 'admit'), axis=1)
        df_transfers['PREV_WARDID'] = df_transfers.apply(lambda row: fill_values(row, 'PREV_WARDID', 'admit', 'admit'),
                                                         axis=1)
        df_transfers['CURR_CAREUNIT'] = df_transfers.apply(
            lambda row: fill_values(row, 'CURR_CAREUNIT', 'discharge', 'discharge'), axis=1)
        df_transfers['CURR_WARDID'] = df_transfers.apply(
            lambda row: fill_values(row, 'CURR_WARDID', 'discharge', 'discharge'), axis=1)

        return df_transfers


    df_transfers = handle_missing_careunit_wardid(df_transfers)

    # make sure intime and outtime are datetime
    df_transfers['INTIME'] = pd.to_datetime(df_transfers['INTIME'])
    df_transfers['OUTTIME'] = pd.to_datetime(df_transfers['OUTTIME'])

    # cumulative timedelta
    df_transfers['cumulative_timedelta'] = df_transfers.groupby('HADM_ID')['OUTTIME'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 3600)

    # transitions and total counts
    df_transfers['careunit_transition'] = (df_transfers['PREV_CAREUNIT'] != df_transfers['CURR_CAREUNIT']).astype(int)
    df_transfers['ward_transition'] = (df_transfers['PREV_WARDID'] != df_transfers['CURR_WARDID']).astype(int)
    df_transfers['total_transfers'] = df_transfers.groupby('HADM_ID')['HADM_ID'].transform('count')
    df_transfers['distinct_careunits_count'] = df_transfers.groupby('HADM_ID')['CURR_CAREUNIT'].transform('nunique')

    # interaction between the careunits
    df_transfers['careunit_interaction'] = df_transfers['PREV_CAREUNIT'] + '_' + df_transfers['CURR_CAREUNIT']

    # mode of specified columns per hadm_id
    df_transfers = make_mode_column(df_transfers, 'PREV_CAREUNIT')
    df_transfers = make_mode_column(df_transfers, 'CURR_CAREUNIT')

    # save
    save_csv(df_transfers, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_transfers.head())

# datetime engineering
section_name = '211_datetime.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_datetime = load_csv(os.path.join(out_dir, section_name))
else:
    df_datetime = replace_nan_with_none(df_datetime, ['ICUSTAY_ID', 'ITEMID', 'LABEL', 'CGID'])

    # make sure CHARTTIME and STORETIME are datetime
    df_datetime['CHARTTIME'] = pd.to_datetime(df_datetime['CHARTTIME'])
    df_datetime['STORETIME'] = pd.to_datetime(df_datetime['STORETIME'])
    df_datetime['time_difference'] = (df_datetime['STORETIME'] - df_datetime['CHARTTIME']).dt.total_seconds() / 3600

    # calculate CGID counts per HADM_ID
    df_datetime['cgid_count_per_hadm_id'] = df_datetime.groupby('HADM_ID')['CGID'].transform('count')

    # calculate modes for ITEMID, LABEL, CGID columns
    df_datetime = make_mode_column(df_datetime, 'ITEMID')
    df_datetime = make_mode_column(df_datetime, 'LABEL')
    df_datetime = make_mode_column(df_datetime, 'CGID')

    # save the dataframe
    save_csv(df_datetime, os.path.join(out_dir, section_name))
    print(df_datetime.head())
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# iecv feature engineering
section_name = '212_iecv.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    df_iecv = load_csv(os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
else:
    # working with df_iecv
    df_iecv = replace_nan_with_none(df_iecv, ['ICUSTAY_ID', 'ITEMID', 'CGID', 'ORIGINALROUTE', 'ORIGINALRATE',
                                              'ORIGINALSITE', 'time_since_admit'])

    # total amount and rate per HADM_ID
    df_iecv['total_amount_per_hadm_id'] = df_iecv.groupby('HADM_ID')['AMOUNT'].transform('sum')
    df_iecv['total_rate_per_hadm_id'] = df_iecv.groupby('HADM_ID')['RATE'].transform('sum')

    # interaction features
    df_iecv['amount_rate_interaction'] = df_iecv['AMOUNT'] * df_iecv['RATE']
    df_iecv['total_amount_rate_interaction'] = df_iecv['total_amount_per_hadm_id'] * df_iecv['total_rate_per_hadm_id']

    # count features
    df_iecv['itemid_count_per_hadm_id'] = df_iecv.groupby('HADM_ID')['ITEMID'].transform('count')

    # calculate modes for ITEMID, LABEL, CGID columns
    df_iecv = make_mode_column(df_iecv, 'ITEMID')
    df_iecv = make_mode_column(df_iecv, 'LABEL')
    df_iecv = make_mode_column(df_iecv, 'CGID')

    # calculate mean and mean_delta for specified columns
    columns_to_calculate = ['AMOUNT', 'RATE', 'ORIGINALAMOUNT', 'total_amount_per_hadm_id', 'total_rate_per_hadm_id',
                            'itemid_count_per_hadm_id']

    for column in columns_to_calculate:
        mean_value = df_iecv[column].mean()
        mean_delta_column = f'mean_delta_{column.lower()}'
        df_iecv[mean_delta_column] = df_iecv[column] - mean_value

    # save the dataframe
    save_csv(df_iecv, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_iecv.head())

# iemv feature engineering
section_name = '213_iemv.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_iemv = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure datetime columns are converted before replacing with 'none provided'
    df_iemv['STARTTIME'] = pd.to_datetime(df_iemv['STARTTIME'])
    df_iemv['ENDTIME'] = pd.to_datetime(df_iemv['ENDTIME'])
    df_iemv['STORETIME'] = pd.to_datetime(df_iemv['STORETIME'])

    # replace all columns except hadm_id and los_hours with 'none provided'
    columns_to_replace = df_iemv.columns.difference(['HADM_ID', 'los_hours']).tolist()
    df_iemv = replace_nan_with_none(df_iemv, columns_to_replace)

    # convert columns that should be numeric
    numeric_columns = ['RATE', 'AMOUNT', 'TOTALAMOUNT', 'PATIENTWEIGHT']
    for col in numeric_columns:
        df_iemv[col] = pd.to_numeric(df_iemv[col], errors='coerce')

    # interaction features
    df_iemv['order_category_interaction'] = df_iemv['ORDERCATEGORYNAME'] + '_' + df_iemv['ORDERCATEGORYDESCRIPTION']
    df_iemv['order_component_interaction'] = df_iemv['ORDERCOMPONENTTYPEDESCRIPTION'] + '_' + df_iemv[
        'ORDERCATEGORYDESCRIPTION']
    df_iemv['total_amount_rate_ratio'] = df_iemv.apply(
        lambda row: row['TOTALAMOUNT'] / row['RATE'] if row['RATE'] != 0 else 0, axis=1)

    # count of occurrences of cgid for each hadm_id
    df_iemv['cgid_count_per_hadm_id'] = df_iemv.groupby('HADM_ID')['CGID'].transform('count')

    # numeric transformations
    df_iemv['average_amount'] = df_iemv.groupby('HADM_ID')['AMOUNT'].transform('mean')
    df_iemv['sum_amount'] = df_iemv.groupby('HADM_ID')['AMOUNT'].transform('sum')
    df_iemv['max_rate'] = df_iemv.groupby('HADM_ID')['RATE'].transform('max')
    df_iemv['rate_to_amount_ratio'] = df_iemv.apply(
        lambda row: row['RATE'] / row['AMOUNT'] if row['AMOUNT'] != 0 else 0, axis=1)

    # save and print
    save_csv(df_iemv, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_iemv.head())

# note events feature engineering
section_name = '214_note_events.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_note = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure datetime columns are converted before feature engineering
    df_note['CHARTTIME'] = pd.to_datetime(df_note['CHARTTIME'])
    df_note['STORETIME'] = pd.to_datetime(df_note['STORETIME'])

    # count of occurrences of cgid for each hadm_id
    df_note['cgid_count'] = df_note.groupby('HADM_ID')['CGID'].transform('count')

    # unique counts of category and description per hadm_id
    df_note['unique_category_count'] = df_note.groupby('HADM_ID')['CATEGORY'].transform('nunique')
    df_note['unique_description_count'] = df_note.groupby('HADM_ID')['DESCRIPTION'].transform('nunique')

    # interactions columns
    df_note['cgid_category_interaction'] = df_note['CGID'].astype(str) + '_' + df_note['CATEGORY']

    # calculate modes for CATEGORY, DESCRIPTION, CGID columns
    df_note = make_mode_column(df_note, 'CATEGORY')
    df_note = make_mode_column(df_note, 'DESCRIPTION')
    df_note = make_mode_column(df_note, 'CGID')

    # substituting with none
    df_note = replace_nan_with_none(df_note, ['CHARTTIME', 'STORETIME', 'CATEGORY', 'CGID', 'time_since_admit'])

    # save the dataframe
    save_csv(df_note, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_note.head())

# output events feature engineering
section_name = '215_output_events.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_output = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure datetime columns are converted before feature engineering
    df_output['CHARTTIME'] = pd.to_datetime(df_output['CHARTTIME'])
    df_output['STORETIME'] = pd.to_datetime(df_output['STORETIME'])

    # substituting with none
    df_output = replace_nan_with_none(df_output, ['CHARTTIME', 'STORETIME', 'LABEL', 'CGID', 'time_since_admit'])

    # count features
    df_output['label_count'] = df_output.groupby('HADM_ID')['LABEL'].transform('count')
    df_output['cgid_count'] = df_output.groupby('HADM_ID')['CGID'].transform('count')

    # calculate modes for LABEL and CGID columns
    df_output = make_mode_column(df_output, 'LABEL')
    df_output = make_mode_column(df_output, 'CGID')

    # calculate mean and median for VALUE column
    df_output['mean_value'] = df_output.groupby('HADM_ID')['VALUE'].transform('mean')
    df_output['median_value'] = df_output.groupby('HADM_ID')['VALUE'].transform('median')

    # save the dataframe
    save_csv(df_output, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_output.head())

# pemv feature engineering
section_name = '216_pemv.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_pemv = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure datetime columns are converted before feature engineering
    df_pemv['STARTTIME'] = pd.to_datetime(df_pemv['STARTTIME'])
    df_pemv['ENDTIME'] = pd.to_datetime(df_pemv['ENDTIME'])

    # replace specified columns with 'None Provided'
    columns_to_replace = ['CGID', 'LABEL', 'ITEMID', 'ORDERCATEGORYNAME', 'ORDERCATEGORYDESCRIPTION', 'LOCATION',
                          'LOCATIONCATEGORY']
    df_pemv = replace_nan_with_none(df_pemv, columns_to_replace)

    # interaction features
    df_pemv['label_ordercategory_interaction'] = df_pemv['LABEL'] + '_' + df_pemv['ORDERCATEGORYNAME']

    # numeric transformations
    df_pemv['sum_value'] = df_pemv.groupby('HADM_ID')['VALUE'].transform('sum')
    df_pemv['average_value'] = df_pemv.groupby('HADM_ID')['VALUE'].transform('mean')
    df_pemv['median_value'] = df_pemv.groupby('HADM_ID')['VALUE'].transform('median')

    # count unique occurrences of CGID, LABEL, LOCATION, ORDERCATEGORYNAME per HADM_ID
    df_pemv['unique_cgid'] = df_pemv.groupby('HADM_ID')['CGID'].transform('nunique')
    df_pemv['unique_label'] = df_pemv.groupby('HADM_ID')['LABEL'].transform('nunique')
    df_pemv['unique_locatio'] = df_pemv.groupby('HADM_ID')['LOCATION'].transform('nunique')
    df_pemv['unique_ordercategoryname'] = df_pemv.groupby('HADM_ID')['ORDERCATEGORYNAME'].transform('nunique')

    # calculate modes for CGID, LABEL, LOCATION, ORDERCATEGORYNAME columns
    df_pemv = make_mode_column(df_pemv, 'CGID')
    df_pemv = make_mode_column(df_pemv, 'LABEL')
    df_pemv = make_mode_column(df_pemv, 'LOCATION')
    df_pemv = make_mode_column(df_pemv, 'ORDERCATEGORYNAME')

    # save the dataframe
    print(df_pemv.head())
    save_csv(df_pemv, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# cpt events feature engineering
section_name = '311_cpt.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    df_cpt = load_csv(os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
else:
    # ensure datetime columns are converted before feature engineering
    df_cpt['CHARTDATE'] = pd.to_datetime(df_cpt['CHARTDATE'])

    # count of unique values per HADM_ID
    df_cpt['unique_subsectionheader'] = df_cpt.groupby('HADM_ID')['SUBSECTIONHEADER'].transform('nunique')
    df_cpt['unique_cpt'] = df_cpt.groupby('HADM_ID')['CPT_CD'].transform('nunique')
    df_cpt['unique_cpt_number'] = df_cpt.groupby('HADM_ID')['CPT_NUMBER'].transform('nunique')

    # replace nan with 'None Provided' for specified columns
    df_cpt = replace_nan_with_none(df_cpt, ['COSTCENTER', 'CHARTDATE', 'CPT_CD', 'CPT_NUMBER', 'SECTIONHEADER',
                                            'SUBSECTIONHEADER'])

    # calculate modes for specified columns per HADM_ID
    df_cpt = make_mode_column(df_cpt, 'COSTCENTER')
    df_cpt = make_mode_column(df_cpt, 'CPT_CD')
    df_cpt = make_mode_column(df_cpt, 'CPT_NUMBER')
    df_cpt = make_mode_column(df_cpt, 'SECTIONHEADER')
    df_cpt = make_mode_column(df_cpt, 'SUBSECTIONHEADER')

    # save
    save_csv(df_cpt, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)
    print(df_cpt.head())

# icd9 diag feature engineer
section_name = '312_icd9d.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_icd9d = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure datetime columns are converted before feature engineering
    df_icd9d = replace_nan_with_none(df_icd9d, ['SEQ_NUM', 'ICD9_CODE', 'diagnosis_category'])

    # unique counts
    df_icd9d['u_icd9_count'] = df_icd9d.groupby('HADM_ID')['ICD9_CODE'].transform('nunique')
    df_icd9d['u_diag_category'] = df_icd9d.groupby('HADM_ID')['diagnosis_category'].transform('nunique')

    # mode of category
    df_icd9d = make_mode_column(df_icd9d, 'diagnosis_category')

    # save the dataframe
    save_csv(df_icd9d, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# drg feature engineer
section_name = '313_drg.csv'

if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
else:
    # ensure drg_severity and drg_mortality are numeric
    df_drg['DRG_SEVERITY'] = pd.to_numeric(df_drg['DRG_SEVERITY'], errors='coerce')
    df_drg['DRG_MORTALITY'] = pd.to_numeric(df_drg['DRG_MORTALITY'], errors='coerce')

    # create new features by combining drg_type and drg_code
    df_drg['drg_type_code'] = df_drg['DRG_TYPE'] + '_' + df_drg['DRG_CODE'].astype(str)

    # interaction between drg_severity and drg_mortality
    df_drg['severity_mortality_interaction'] = df_drg['DRG_SEVERITY'] * df_drg['DRG_MORTALITY']

    # mean, sum, and max of drg_severity and drg_mortality for each hadm_id
    df_drg['mean_drg_severity'] = df_drg.groupby('HADM_ID')['DRG_SEVERITY'].transform('mean')
    df_drg['sum_drg_severity'] = df_drg.groupby('HADM_ID')['DRG_SEVERITY'].transform('sum')
    df_drg['max_drg_severity'] = df_drg.groupby('HADM_ID')['DRG_SEVERITY'].transform('max')
    df_drg['mean_drg_mortality'] = df_drg.groupby('HADM_ID')['DRG_MORTALITY'].transform('mean')
    df_drg['sum_drg_mortality'] = df_drg.groupby('HADM_ID')['DRG_MORTALITY'].transform('sum')
    df_drg['max_drg_mortality'] = df_drg.groupby('HADM_ID')['DRG_MORTALITY'].transform('max')

    # cumulative sum of drg_severity and drg_mortality for each hadm_id
    df_drg['cumulative_drg_severity'] = df_drg.groupby('HADM_ID')['DRG_SEVERITY'].cumsum()
    df_drg['cumulative_drg_mortality'] = df_drg.groupby('HADM_ID')['DRG_MORTALITY'].cumsum()

    # count the occurrences of each drg_type and drg_code for each hadm_id
    df_drg['drg_type_count'] = df_drg.groupby('HADM_ID')['DRG_TYPE'].transform('count')
    df_drg['drg_code_count'] = df_drg.groupby('HADM_ID')['DRG_CODE'].transform('count')

    # replace nan with 'none provided' for severity and mortality columns
    df_drg = replace_nan_with_none(df_drg, ['DRG_SEVERITY', 'DRG_MORTALITY'])

    # save the dataframe
    save_csv(df_drg, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# labs feature engineering
section_name = '314_labs.csv'
if os.path.isfile(os.path.join(out_dir, section_name)):
    df_labs = load_csv(os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
else:
    # replace nan with 'none provided' for FLAG
    df_labs = replace_nan_with_none(df_labs, ['FLAG'])

    # this table has a value that is not numeric, so we need to remove it from the numeric processing for correlation
    # stat purposes. This is a quick and dirty way of doing that, I hope
    df_labs['LABSVALUE'] = df_labs['VALUENUM']

    # drop the LABEL column as it's totally empty
    df_labs = df_labs.drop(columns=['LABEL'])
    df_labs = df_labs.drop(columns=['VALUENUM'])

    # unique counts
    df_labs['unique_itemid_count'] = df_labs.groupby('HADM_ID')['ITEMID'].transform('nunique')
    df_labs['unique_value_count'] = df_labs.groupby('HADM_ID')['LABSVALUE'].transform('nunique')
    df_labs['unique_flag_count'] = df_labs.groupby('HADM_ID')['FLAG'].transform('nunique')

    # calculate mean LABSVALUE per HADM_ID
    df_labs['mean_labsvalue'] = df_labs.groupby('HADM_ID')['LABSVALUE'].transform('mean')

    # calculate mode of ITEMID per HADM_ID
    df_labs = make_mode_column(df_labs, 'ITEMID')

    # save the dataframe
    save_csv(df_labs, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# microbiology feature engineering
section_name = '315_microbiology.csv'
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_microbiology = load_csv(os.path.join(out_dir, section_name))
else:
    # replace nan with 'none provided' for all columns except dilution value and time since admit, hadmid and los hours
    columns_to_replace = df_microbiology.columns.difference(
        ['DILUTION_VALUE', 'time_since_admit', 'HADM_ID', 'los_hours']).tolist()
    df_microbiology = replace_nan_with_none(df_microbiology, columns_to_replace)

    # unique counts
    df_microbiology['unique_spec_type_desc'] = df_microbiology.groupby('HADM_ID')['SPEC_TYPE_DESC'].transform('nunique')
    df_microbiology['unique_isolate_num'] = df_microbiology.groupby('HADM_ID')['ISOLATE_NUM'].transform('nunique')
    df_microbiology['unique_org_itemid'] = df_microbiology.groupby('HADM_ID')['ORG_ITEMID'].transform('nunique')
    df_microbiology['unique_ab_itemid'] = df_microbiology.groupby('HADM_ID')['AB_ITEMID'].transform('nunique')

    # total counts
    df_microbiology['total_spec_type_desc'] = df_microbiology.groupby('HADM_ID')['SPEC_TYPE_DESC'].transform('count')
    df_microbiology['total_isolate_num'] = df_microbiology.groupby('HADM_ID')['ISOLATE_NUM'].transform('count')
    df_microbiology['total_org_itemid'] = df_microbiology.groupby('HADM_ID')['ORG_ITEMID'].transform('count')
    df_microbiology['total_ab_itemid'] = df_microbiology.groupby('HADM_ID')['AB_ITEMID'].transform('count')

    # interaction feature
    df_microbiology['org_itemid_interpretation'] = \
        (df_microbiology['ORG_ITEMID'].astype(str) + '_' + df_microbiology['INTERPRETATION'])

    # save
    save_csv(df_microbiology, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# prescriptions feature engineering
section_name = '316_prescriptions.csv'
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_prescriptions = load_csv(os.path.join(out_dir, section_name))
else:
    # ensure ndc is treated as a string and dose val rx is a number
    df_prescriptions['NDC'] = df_prescriptions['NDC'].astype(str)
    df_prescriptions['DOSE_VAL_RX'] = pd.to_numeric(df_prescriptions['DOSE_VAL_RX'], errors='coerce')

    # uniques
    df_prescriptions['unique_ndc'] = df_prescriptions.groupby('HADM_ID')['NDC'].transform('nunique')
    df_prescriptions['unique_route_per_hadm_id'] = df_prescriptions.groupby('HADM_ID')['ROUTE'].transform('nunique')
    df_prescriptions['total_ndc'] = df_prescriptions.groupby('HADM_ID')['NDC'].transform('count')

    # calculate total, mean, and max of dose_val_rx per hadm_id
    df_prescriptions['total_dose_val_rx'] = df_prescriptions.groupby('HADM_ID')['DOSE_VAL_RX'].transform('sum')
    df_prescriptions['mean_dose_val_rx'] = df_prescriptions.groupby('HADM_ID')['DOSE_VAL_RX'].transform('mean')
    df_prescriptions['max_dose_val_rx'] = df_prescriptions.groupby('HADM_ID')['DOSE_VAL_RX'].transform('max')
    df_prescriptions['cumulative_dose_val_rx'] = df_prescriptions.groupby('HADM_ID')['DOSE_VAL_RX'].cumsum()

    # calculate total and unique drug counts per hadm_id
    df_prescriptions['total_drug'] = df_prescriptions.groupby('HADM_ID')['DRUG'].transform('count')
    df_prescriptions['unique_drug'] = df_prescriptions.groupby('HADM_ID')['DRUG'].transform('nunique')

    # calculate total and unique gsn counts per hadm_id
    df_prescriptions['total_gsn'] = df_prescriptions.groupby('HADM_ID')['GSN'].transform('count')
    df_prescriptions['unique_gsn'] = df_prescriptions.groupby('HADM_ID')['GSN'].transform('nunique')

    # drop the NDC and GSN columns
    df_prescriptions = df_prescriptions.drop(columns=['NDC', 'GSN'])

    # save the dataframe
    save_csv(df_prescriptions, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

# procedures feature engineer
section_name = '317_procedures.csv'
if os.path.isfile(os.path.join(out_dir, section_name)):
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} found in folder, skipping creation in {time_elapsed}', log_path)
    df_procedures = load_csv(os.path.join(out_dir, section_name))
else:
    # calculate uniques
    df_procedures['u_icd9_codes'] = df_procedures.groupby('HADM_ID')['ICD9_CODE'].transform('nunique')
    df_procedures['u_icd9_categories'] = df_procedures.groupby('HADM_ID')['icd9_procedure_category'].transform(
        'nunique')

    # replace nan with 'none provided' for icd9_procedure_category
    df_procedures = replace_nan_with_none(df_procedures, ['ICD9_CODE', 'icd9_procedure_category'])

    # calculate mode of icd9_procedure_category per HADM_ID
    df_procedures = make_mode_column(df_procedures, 'icd9_procedure_category')

    # save the dataframe
    save_csv(df_procedures, os.path.join(out_dir, section_name))
    time_elapsed = round(time.time() - st_time, 3)
    log_output(f'{section_name} saved at {time_elapsed}', log_path)

'''
step 3.5
statistical analysis for the new feature engineered columns
'''

# refresh dataframe references
dataframes = [
    ('df_admissions', df_admissions),
    ('df_callout', df_callout),
    ('df_icustays', df_icustays),
    ('df_services', df_services),
    ('df_transfers', df_transfers),
    ('df_datetime', df_datetime),
    ('df_iecv', df_iecv),
    ('df_iemv', df_iemv),
    ('df_note', df_note),
    ('df_output', df_output),
    ('df_pemv', df_pemv),
    ('df_cpt', df_cpt),
    ('df_icd9d', df_icd9d),
    ('df_drg', df_drg),
    ('df_labs', df_labs),
    ('df_microbiology', df_microbiology),
    ('df_prescriptions', df_prescriptions),
    ('df_procedures', df_procedures)
]

# specify output file paths
fe_cat_sig_path = os.path.join(out_dir, '013_fe_categorical_significant.csv')
fe_num_sig_path = os.path.join(out_dir, '014_fe_numerical_significant.csv')
correlation_cat_path = os.path.join(out_dir, '015_categorical_correlation.csv')
correlation_num_path = os.path.join(out_dir, '016_numerical_correlation.csv')

# check if output files exist
if all(os.path.exists(path) for path in [fe_cat_sig_path, fe_num_sig_path, correlation_cat_path, correlation_num_path]):
    print('Output files found, loading factors from existing files.')

    # load the dataframes from the existing files
    fe_cat_df_sig = pd.read_csv(fe_cat_sig_path)
    fe_num_df_sig = pd.read_csv(fe_num_sig_path)
    correlation_categorical_results_df = pd.read_csv(correlation_cat_path)
    correlation_numerical_results_df = pd.read_csv(correlation_num_path)

    # print the loaded dataframes
    print('\nCorrelation Categorical Results:')
    print(fe_cat_df_sig)

    print('\nCorrelation Numerical Results:')
    print(fe_num_df_sig)
else:
    # perform the analysis as normal if the files are not found
    categorical_results_fe = []
    numerical_results_fe = []

    # setting variables for statistical analysis
    target = 'los_hours'

    # these are settings for the statistical and correlation analysis
    forced_categorical = ['ICD9_CODE', 'DRG_CODE', 'NDC', 'CGID', 'PREV_WARDID']

    forced_numerical = [
        'ed_los', 'timedelta', 'icu_los', 'average_icu_los', 'total_icu_los', 'total_transfers', 'VALUE',
        'time_since_admit', 'cgid_count_per_hadm_id', 'AMOUNT', 'RATE', 'num_services', 'ORIGINALAMOUNT',
        'total_amount_per_hadm_id', 'total_rate_per_hadm_id', 'itemid_count_per_hadm_id', 'cgid_count_per_hadm_id',
        'AMOUNT', 'RATE', 'TOTALAMOUNT', 'PATIENTWEIGHT', 'time_since_admit', 'cgid_count_per_hadm_id',
        'average_amount_per_hadm_id', 'sum_amount_per_hadm_id', 'max_rate_per_hadm_id', 'rate_to_amount_ratio',
        'time_since_admit', 'VALUE', 'time_since_admit', 'VALUE', 'time_since_admit', 'sum_value_per_hadm_id',
        'average_value_per_hadm_id', 'time_since_admit', 'SEQ_NUM', 'u_icd9_count', 'u_diag_category',
        'severity_mortality_interaction', 'mean_drg_severity', 'sum_drg_severity', 'max_drg_severity',
        'mean_drg_mortality', 'sum_drg_mortality', 'max_drg_mortality', 'cumulative_drg_severity',
        'cumulative_drg_mortality', 'drg_type_count', 'drg_code_count', 'VALUENUM', 'time_since_admit',
        'time_since_admit', 'DOSE_VAL_RX', 'time_since_admit', 'unique_ndc', 'total_ndc',
        'total_dose_val_rx', 'max_dose_val_rx', 'cumulative_dose_val_rx', 'SEQ_NUM', 'u_icd9_codes',
        'u_icd9_categories', 'time_difference', 'cgid_count_per_hadm_id', 'run_time', 'unique_itemid_count',
        'unique_value_count', 'unique_flag_count', 'unique_spec_type_desc', 'unique_isolate_num',
        'unique_org_itemid', 'unique_ab_itemid', 'total_spec_type_desc', 'total_isolate_num',
        'total_org_itemid', 'total_ab_itemid', 'unique_route_per_hadm_id', 'total_drug', 'unique_drug', 'total_gsn',
        'unique_gsn']

    exclusions = ['HADM_ID', 'SUBJECT_ID', 'los_hours', 'ADMITTIME', 'STARTDATE', 'ENDDATE', 'STORETIME',
                  'VALUEUOM', 'OUTCOMETIME', 'ICUSTAY_ID', 'TRANSFERTIME', 'STARTTIME', 'ENDTIME',
                  'EDREGTIME', 'EDOUTTIME', 'CHARTTIME', 'CHARTDATE', 'INTIME', 'OUTTIME', 'CREATETIME', 'DISCHTIME',
                  'seq_num', 'SEQ_NUM']

    for df_name, df in dataframes:
        print(f'processing correlation results for dataframe: {df_name}')
        categorical_results_df_correlation, numerical_results_df_correlation = outlier_stats_output_2(
            df, target=target, original_df_name=df_name, exclusions=exclusions, forced_categorical=forced_categorical,
            forced_numerical=forced_numerical)
        categorical_results_fe.append(categorical_results_df_correlation)
        numerical_results_fe.append(numerical_results_df_correlation)

    # concatenate correlation results
    correlation_categorical_results_df = pd.concat(categorical_results_fe, ignore_index=True).sort_values(by='kw_stat',
                                                                                                          ascending=False)
    correlation_numerical_results_df = pd.concat(numerical_results_fe, ignore_index=True).sort_values(by='spearman_cc',
                                                                                                      ascending=False)

    fe_cat_df_sig = correlation_categorical_results_df[correlation_categorical_results_df['sig_kw'] == 1]
    fe_num_df_sig = correlation_numerical_results_df[correlation_numerical_results_df['sig_spearman'] == 1]

    # ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save the dataframes
    fe_cat_df_sig.to_csv(fe_cat_sig_path, index=False)
    fe_num_df_sig.to_csv(fe_num_sig_path, index=False)
    correlation_categorical_results_df.to_csv(correlation_cat_path, index=False)
    correlation_numerical_results_df.to_csv(correlation_num_path, index=False)

    # print the results
    print('\nCorrelation Categorical Results:')
    print(fe_cat_df_sig)

    print('\nCorrelation Numerical Results:')
    print(fe_num_df_sig)

# create lists of factors from significant results
cat_factors = fe_cat_df_sig['factor'].tolist()
num_factors = fe_num_df_sig['factor'].tolist()

# print the factors
print('Categorical Factors:')
print(cat_factors)

print('Numerical Factors:')
print(num_factors)

exclusions = ['HADM_ID', 'SUBJECT_ID', 'los_hours', 'ADMITTIME', 'STARTDATE', 'ENDDATE', 'STORETIME', 'VALUEUOM',
              'OUTCOMETIME', 'ICUSTAY_ID', 'TRANSFERTIME', 'STARTTIME', 'ENDTIME', 'EDREGTIME', 'EDOUTTIME',
              'CHARTTIME', 'CHARTDATE', 'INTIME', 'OUTTIME', 'CREATETIME', 'DISCHTIME']

# define the output directory correctly
out_dir = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\output_files'

'''
Step 4:
data synthesis.

This is where we synthesize the data for the machine learning part of the project. 

'''

# synthesizing numerical columns in preparation for machine learning
for df_name, df in dataframes:
    synth_file_path = os.path.join(out_dir, f'{df_name}_synth.csv')

    if os.path.exists(synth_file_path):
        print(f'{synth_file_path} already exists, loading data.')
        df_synth = load_csv2(synth_file_path)
    else:
        if df_name == 'df_datetime':
            print(f'Skipping imputation for dataframe: {df_name}')
            continue

        print(f'Imputing missing values for dataframe: {df_name}')

        # filter numerical factors to exclude columns in exclusions
        num_factors_in_df = [factor for factor in num_factors if factor in df.columns and factor not in exclusions]

        if num_factors_in_df:
            # convert numeric columns to numbers (you can never be too sure)
            for col in num_factors_in_df:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # apply the stochastic imputation
            for col in num_factors_in_df:
                df[col] = stochastic_imputation(df, col)

        # find the completeness of numerical columns
        completeness_func(df[num_factors_in_df])

        # save the dataframe
        save_csv(df, synth_file_path)
        print(f'Saved imputed dataframe to {synth_file_path}')

# redeclare dataframes to load the synthetic versions (except datetime cos it was a nuisance.)
df_ref = load_csv2('000_reference_data.csv')
df_admissions = load_csv2('df_admissions_synth.csv')
df_callout = load_csv2('df_callout_synth.csv')
df_icustays = load_csv2('df_icustays_synth.csv')
df_services = load_csv2('df_services_synth.csv')
df_transfers = load_csv2('df_transfers_synth.csv')
df_datetime = load_csv2('211_datetime.csv')
df_iecv = load_csv2('df_iecv_synth.csv')
df_iemv = load_csv2('df_iemv_synth.csv')
df_note = load_csv2('df_note_synth.csv')
df_output = load_csv2('df_output_synth.csv')
df_pemv = load_csv2('df_pemv_synth.csv')
df_cpt = load_csv2('df_cpt_synth.csv')
df_icd9d = load_csv2('df_icd9d_synth.csv')
df_drg = load_csv2('df_drg_synth.csv')
df_labs = load_csv2('df_labs_synth.csv')
df_microbiology = load_csv2('df_microbiology_synth.csv')
df_prescriptions = load_csv2('df_prescriptions_synth.csv')
df_procedures = load_csv2('df_procedures_synth.csv')

dataframes = [
    ('df_admissions', df_admissions),
    ('df_callout', df_callout),
    ('df_icustays', df_icustays),
    ('df_services', df_services),
    ('df_transfers', df_transfers),
    ('df_datetime', df_datetime),
    ('df_iecv', df_iecv),
    ('df_iemv', df_iemv),
    ('df_note', df_note),
    ('df_output', df_output),
    ('df_pemv', df_pemv),
    ('df_cpt', df_cpt),
    ('df_icd9d', df_icd9d),
    ('df_drg', df_drg),
    ('df_labs', df_labs),
    ('df_microbiology', df_microbiology),
    ('df_prescriptions', df_prescriptions),
    ('df_procedures', df_procedures)
]

# loading in this dataframe to find out which columns are significant.
df_num_sig = load_csv2('014_fe_numerical_significant.csv')
df_cat_sig = load_csv2('013_fe_categorical_significant.csv')

# define the target directory
target_directory = 'C:/Users/ander/Documents/.Uni/Project/mimic-iii-clinical-database-1.4/.unpacked/'

# get the list of significant columns
num_sig_columns = df_num_sig['factor'].tolist()
cat_sig_columns = df_cat_sig['factor'].tolist()

significant_columns = num_sig_columns + cat_sig_columns

# loop through dataframes, drop rows, filter columns, expand and combine them
combined_df = pd.DataFrame()
all_dropped_columns = []

for tw in time_window:
    for ne in num_entries:

        combined_df = pd.DataFrame()
        all_dropped_columns = []

        for df_name, df in dataframes:
            df = drop_time_window(df, tw)  # drop rows based on time window first
            df = make_max_sequence_num(df, df_name)  # create max_sequence_num column

            # only keep the original sequence number columns
            original_seq_num_cols = [col for col in df.columns if 'seq_num' in col.lower() and '_dev' not in col]
            other_columns = [col for col in df.columns if
                             'seq_num' not in col.lower() and '_dev' not in col and col != 'HADM_ID']

            df = make_total_mean_seq(df[original_seq_num_cols + other_columns + ['HADM_ID']], df_name)

            df = make_total_mean_hours(df, df_name)
            df = drop_rows_seq_num(df, df_name, ne)

            df, dropped_columns = filter_columns(df, significant_columns)
            all_dropped_columns.extend(dropped_columns)
            transformed_df = expand_rows_to_cols(df, df_name, exp_exceptions)
            if combined_df.empty:
                combined_df = transformed_df
            else:
                combined_df = pd.merge(combined_df, transformed_df, on='HADM_ID', how='outer')

        # remove duplicate columns based on content
        combined_df = remove_duplicate_columns(combined_df)

        # apply stochastic imputation only to numeric columns
        for col in combined_df.columns:
            if pd.api.types.is_numeric_dtype(combined_df[col]):
                print(f'Imputing missing values for column: {col}')
                combined_df[col] = stochastic_imputation(combined_df, col)

        # dfind completeness of the combined dataframe
        completeness_df = completeness_func(combined_df)

        # do mode imputation on columns with completeness above the threshold
        mode_imputer = SimpleImputer(strategy='most_frequent')
        for col in completeness_df[completeness_df['Completeness (%)'] >= completeness_threshold]['column']:
            print(f'Applying mode imputation to column: {col}')
            combined_df[col] = mode_imputer.fit_transform(combined_df[[col]]).ravel()

        # then fixed value imputation to columns with completeness below the threshold
        fixed_value_imputer = SimpleImputer(strategy='constant', fill_value='none')
        for col in completeness_df[completeness_df['Completeness (%)'] < completeness_threshold]['column']:
            if combined_df[col].dtype == 'object':
                print(f'Applying fixed value imputation to column: {col}')
                combined_df[col] = fixed_value_imputer.fit_transform(combined_df[[col]]).ravel()
            else:
                print(f'Skipping fixed value imputation for column: {col} because its dtype is not object.')

        # find overall completeness of the combined dataframe
        total_cells = combined_df.size
        missing_cells = combined_df.isna().sum().sum()
        overall_completeness = ((total_cells - missing_cells) / total_cells) * 100

        # log overall completeness
        log_output(f'\nOverall completeness of the combined dataframe: {overall_completeness:.2f}%', log_path)

        # log the head and a sample of the combined dataframe
        log_output('Head of the combined dataframe:', log_path)
        log_output(combined_df.head(10).to_string(index=False), log_path)

        # check the number of unique HADM_IDs
        unique_hadm_ids = combined_df['HADM_ID'].nunique()

        # check the total number of rows
        total_rows = combined_df.shape[0]

        # log the comparison
        log_output(f'Number of unique HADM_IDs: {unique_hadm_ids}', log_path)
        log_output(f'Total number of rows: {total_rows}', log_path)


        # function to classify stay based on los_hours
        def classify_stay(los_hours):
            if los_hours <= shortstay:
                return 0
            elif los_hours <= longstay:
                return 1
            else:
                return 2


        # ensure that the hadm_id columns are the same type before merging (CRITICAL)
        combined_df['HADM_ID'] = combined_df['HADM_ID'].astype(str)
        df_ref['HADM_ID'] = df_ref['HADM_ID'].astype(str)

        # merge los_hours from df_ref if it's missing in combined_df
        if 'los_hours' not in combined_df.columns:
            combined_df = pd.merge(combined_df, df_ref[['HADM_ID', 'los_hours']], on='HADM_ID', how='left')

            # insert los_hours right after HADM_ID
            if 'los_hours' in combined_df.columns:
                hadm_id_idx = combined_df.columns.get_loc('HADM_ID')
                los_hours_col = combined_df.pop('los_hours')
                combined_df.insert(hadm_id_idx + 1, 'los_hours', los_hours_col)

                # add los_days right after los_hours
                combined_df['los_days'] = combined_df['los_hours'] / 24
                los_days_col = combined_df.pop('los_days')
                combined_df.insert(hadm_id_idx + 2, 'los_days', los_days_col)

        # add a new column for stay classification
        combined_df['stay_class'] = combined_df['los_hours'].apply(classify_stay)

        # print the dropped columns before saving
        print(f'Dropped columns: {set(all_dropped_columns)}')

        print(combined_df.head())

        # save the final combined dataframe
        save_csv(combined_df, os.path.join(out_dir, f'900_complete_data_synth{tw}_{ne}.csv'))

# list of dataframe variables to clear
dataframes_to_clear = [
    'df_ref', 'df_admissions', 'df_callout', 'df_icustays', 'df_services', 'df_transfers',
    'df_datetime', 'df_iecv', 'df_iemv', 'df_note', 'df_output', 'df_pemv', 'df_cpt',
    'df_icd9d', 'df_drg', 'df_labs', 'df_microbiology', 'df_prescriptions', 'df_procedures',
    'df_num_sig', 'df_cat_sig', 'combined_df', 'completeness_df'
]

# clear the specified dataframes
clear_dataframes(dataframes_to_clear)

# finishing up the dataframes.
exp_exceptions = ['HADM_ID', 'los_hours', 'los_days', 'stay_class']

cat_exceptions = []
specific_columns_to_drop = ['datetime_VALUE']

# set up columns and extra data
df_num_sig = load_csv2('014_fe_numerical_significant.csv')
df_cat_sig = load_csv2('013_fe_categorical_significant.csv')

df_num = load_csv2('016_numerical_correlation.csv')
df_cat = load_csv2('015_categorical_correlation.csv')

num_columns = df_num['factor'].tolist()
cat_columns = df_cat['factor'].tolist()

# managing the numerical and categorical columns and extras which are not classified correctly
additional_numeric_keywords = ['sequence_num', 'seq_num', 'unique', 'median', 'mean', 'total']
num_columns.extend(additional_numeric_keywords)

num_sig_columns = df_num_sig['factor'].tolist()
cat_sig_columns = df_cat_sig['factor'].tolist()

significant_columns = num_sig_columns + cat_sig_columns

# iterate over num_entries, time_window, and max_cat values
for ne in num_entries:
    for tw in time_window:
        for mc in max_cat:
            print(f'processing num_entries={ne}, time_window={tw}, max_cat={mc}')

            # load the data for processing
            combined_df = load_csv2(f'900_complete_data_synth{tw}_{ne}.csv')
            if combined_df is None:
                continue

            combined_cols = combined_df.columns.tolist()

            # check for columns in combined_cols that aren't in num_columns, cat_columns, or exceptions and drop them
            valid_columns = num_columns + cat_columns + exp_exceptions
            columns_to_drop = [col for col in combined_cols if not any(substring in col for substring in valid_columns)]

            # add specific columns to drop to the list, checking for substrings
            for col in combined_cols:
                if any(substring in col for substring in specific_columns_to_drop):
                    columns_to_drop.append(col)

            if columns_to_drop:
                print('dropping columns not in num_columns, cat_columns, exceptions, or specified columns to drop:')
                print(columns_to_drop)
                combined_df.drop(columns=columns_to_drop, inplace=True)

            # convert integer columns to float
            for col in combined_df.select_dtypes(include=['int']).columns:
                combined_df[col] = combined_df[col].astype(float)

            # apply stochastic imputation to float columns except for specific exemptions
            for col in combined_df.columns:
                if pd.api.types.is_float_dtype(combined_df[col]) and col not in exp_exceptions:
                    print(f'imputing missing values for column: {col}')
                    combined_df[col] = stochastic_imputation(combined_df, col)

            # create a DataFrame to hold column name, dtype, % of missing values, and number of unique entries
            missing_data_summary = pd.DataFrame({
                'column_name': combined_df.columns,
                'dtype': combined_df.dtypes,
                'missing_percentage': combined_df.isna().mean() * 100,
                'n_uniques': combined_df.nunique()
            }).reset_index(drop=True)

            # drop columns with more than 80% missing values
            cols_to_drop = missing_data_summary[missing_data_summary['missing_percentage'] > completeness_threshold][
                'column_name']
            combined_df.drop(columns=cols_to_drop, inplace=True)

            # drop categorical columns with more unique entries than max_cat, excluding exceptions
            if mc > 0:
                for col in combined_df.columns:
                    if any(substring in col for substring in cat_columns) and col not in cat_exceptions:
                        if combined_df[col].nunique() > mc:
                            print(f'Dropping column {col} with {combined_df[col].nunique()} unique values')
                            combined_df.drop(columns=[col], inplace=True)

            # recreate missing data summary after dropping columns
            missing_data_summary = pd.DataFrame({
                'column_name': combined_df.columns,
                'dtype': combined_df.dtypes,
                'missing_percentage': combined_df.isna().mean() * 100,
                'n_uniques': combined_df.nunique()
            }).reset_index(drop=True)

            # sort the DataFrame by dtype, placing numeric types at the top
            dtype_order = {'int64': 1, 'float64': 1, 'object': 2}
            missing_data_summary['dtype_order'] = missing_data_summary['dtype'].map(
                lambda x: dtype_order.get(str(x), 3))
            missing_data_summary = missing_data_summary.sort_values(by='dtype_order').drop(
                columns='dtype_order').reset_index(drop=True)

            # print the final missing data summary
            print(missing_data_summary)
            print(combined_df.head())

            # save the cleaned data
            save_csv(combined_df, os.path.join(out_dir, f'901_cleaned_combined_data{tw}_{ne}_{mc}.csv'))

# calculate and log the execution time
end_time = time.time()
elapsed_time = end_time - st_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
log_output(f'Script took {elapsed_time:.2f} seconds to execute.', log_path)
log_output(f'That\'s {int(hours)}:{int(minutes)}:{int(seconds)} (hh:mm:ss). :)', log_path)
