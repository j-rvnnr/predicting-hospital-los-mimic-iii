import gc
import time
import os
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# target directory. Set this to wherever the mimic csv files are kept
target_dir = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked'
sub_dir = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\output_files'
os.chdir(target_dir)
print(f'Directory set to: {target_dir}')

# setting pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# load csv function again, but this time the subdirectory is always output_files
def load_csv2(file_name, subdirectory='output_files'):
    file_path = os.path.join(target_dir, subdirectory, file_name)
    print(f'loading {file_path}')
    return pd.read_csv(file_path, low_memory=False)


# function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# making and setting the output directory for the plots
plot_dir = create_directory('data_exploration')
out_dir = os.path.join(sub_dir, plot_dir)
print(f'Output directory: {out_dir}')
create_directory(out_dir)

# load in some data
df_ref = load_csv2('000_reference_data.csv')
df_core = load_csv2('001_core_data.csv')

df_ref['los_days'] = (df_ref['los_hours'] / 24).round().astype(int)

print(df_ref['los_days'].head())

df_admissions = load_csv2('111_admissions.csv')
df_admissions_c = df_admissions[df_admissions['HADM_ID'].isin(df_core['HADM_ID'])]



df_icustays = load_csv2('103_icu_stays.csv')
#df_iecv = load_csv2('202_iecv.csv')
#df_iemv = load_csv2('203_iemv.csv')

#df_icd9d = load_csv2('302_icd9d.csv')

#df_prescriptions = load_csv2('306_prescriptions.csv')
#df_procedures = load_csv2('307_procedures.csv')

print(df_admissions.head())
print(df_icustays.head())

#print(df_iemv.head())

#print(df_icd9d.head())
#(df_prescriptions.head())
#print(df_procedures.head())





# smooth histogram plot function
def smooth_histogram_plot(df, column, output_dir):
    plt.figure(figsize=(12, 8))
    sns.kdeplot(df[column], fill=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

    # ensure the directory exists
    create_directory(output_dir)

    # save the plot
    plot_path = os.path.join(output_dir, f'kde_{column}.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

    # show the plot
    plt.show()
    plt.close()

# box whisker plot function
def box_whisker_plot(df, column, output_dir, order=None):
    # if no order is specified, rank by median value
    if order is None:
        order = df.groupby(column)['los_hours'].median().sort_values().index.tolist()

    # ensure order only contains values present in the column
    order = [val for val in order if val in df[column].unique()]

    plt.figure(figsize=(12, 8))

    # this line throws a deprecation warning, but I can't figure out how to make it work 'correctly' so it's just going
    # to have to stay there for now
    sns.boxplot(x='los_hours', y=column, data=df, order=order, showfliers=False)

    plt.title(f'Box Whisker Plot of {column} vs. LOS Hours')
    plt.xlabel('LOS Hours')
    plt.ylabel(column)
    plt.xticks(rotation=45)

    # ensure the directory exists
    create_directory(output_dir)

    # save the plot
    plot_path = os.path.join(output_dir, f'box_whisker_{column}.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

    # show the plot
    plt.show()
    plt.close()

# histogram, specifically binned by days
def histogram_plot_days(df, column, output_dir, bin_width=None, num_bins=None):
    # calculate los_days
    df.loc[:, 'los_days'] = df[column] / 24


    max_days = np.ceil(df['los_days'].max())
    if bin_width is not None:
        bins = np.arange(0, max_days + bin_width, bin_width)
    elif num_bins is not None:
        bins = np.linspace(0, max_days, num_bins + 1)
    else:
        bins = np.arange(0, max_days + 1)  # Default to binning by day

    # plot the histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(df['los_days'], bins=bins, kde=False)
    plt.title(f'Histogram of {column} (binned by days)')
    plt.xlabel('LOS Days')
    plt.ylabel('Frequency')

    # adjust x-axis ticks to align with the bins
    plt.xticks(bins)

    # save
    create_directory(output_dir)
    plot_path = os.path.join(output_dir, f'histogram_{column}_by_day.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

    # show
    plt.show()
    plt.close()

def histogram_plot(df, column, output_dir, num_bins=None):

    df['los_days'] = df[column].astype(int)

    # determine the range of bins
    if num_bins is not None:
        max_days = min(df['los_days'].max(), num_bins)
        bins = np.arange(0, max_days + 1)
    else:
        max_days = df['los_days'].max()
        bins = np.arange(0, max_days + 1)  # default to binning by day

    # plot the histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(df['los_days'], bins=bins, kde=False)
    plt.title(f'Histogram of Length of Stay (days)')
    plt.xlabel('LOS days')
    plt.ylabel('Frequency')


    plt.xticks(bins)

    # save
    create_directory(output_dir)
    plot_path = os.path.join(output_dir, f'histogram_{column}_by_day.png')
    plt.savefig(plot_path)
    print(f'plot saved to {plot_path}')

    # sshow
    plt.show()
    plt.close()

histogram_plot(df_ref, 'los_days', out_dir, num_bins=30
               )


# scatter plot function
def scatter_plot(df, column, output_dir):
    plt.figure(figsize=(12, 8))

    # create scatter plot
    plt.scatter(df['los_hours'], df[column], alpha=0.5)

    plt.title(f'Scatter Plot of {column} vs. LOS Hours')
    plt.xlabel('LOS Hours')
    plt.ylabel(column)
    plt.xticks(rotation=45)

    # directory catch
    create_directory(output_dir)

    # save the plot
    plot_path = os.path.join(output_dir, f'scatter_plot_{column}.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

    # show the plot
    plt.show()
    plt.close()









age_range_order = ['16-18', '19-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']






# smooth_histogram_plot(df_admissions_c, 'los_hours', out_dir)
# histogram_plot_days(df_ref, 'los_hours', out_dir, num_bins=30)
# box_whisker_plot(df_admissions_c, 'age_range', out_dir, order=age_range_order)
# box_whisker_plot(df_admissions_c, 'GENDER', out_dir)
# box_whisker_plot(df_admissions_c, 'religion_group', out_dir)
# box_whisker_plot(df_admissions_c, 'insurance_type', out_dir)

# box_whisker_plot(df_admissions_c, 'marital_status_group', out_dir)

#box_whisker_plot(df_icustays, 'FIRST_CAREUNIT', out_dir)

#box_whisker_plot(df_icd9d, 'diagnosis_category', out_dir)

#scatter_plot(df_iemv, 'PATIENTWEIGHT', out_dir)
#scatter_plot(df_icustays, 'icu_los', out_dir)