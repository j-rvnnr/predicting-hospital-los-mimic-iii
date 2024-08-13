import gc
import time
import os
from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Input
from tensorflow.keras.optimizers import AdamW
from ann_visualizer.visualize import ann_viz
from tensorflow.keras.utils import plot_model

####################################################################################################
#                                                                                                  #
# Step 0:                                                                                          #
# Initialise variables                                                                             #
#                                                                                                  #
# These are grouped roughly by the order that they appear in the script, although it's not         #
# perfect. There are clarifying comments to help with some of the variables here.                  #
#                                                                                                  #
####################################################################################################

# setting pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# set the num_only parameter
num_only = 1
calc_feature_import = 0

# define our file paths
target_hours = 72
target_entries = 5
target_catlimit = 20

# these are hyperparameters for the neural network, including runtime, learningrate, batch size and the train/test
# split size.
epochs = 250
learningrate = 0.0005
batchsize = 128
testsize = 0.25 # IT HAS TO BE 0.25 IN THIS SCRIPT OTHERWISE WE CAN GET ERRORS DO NOT CHANGE

# earlystopping parameters
earlystopping_start = 100
earlystopping_patience = 50

# total time limit (seconds)[3600 is 1 hour]
total_time_limit = 3600

# neural network parameters
max_layersize = 512
l1reg = 0.001
l2reg = 0.001
dropout = 0.5

# misc parameters
random_state = 117
top_n = 50

# define stay classification thresholds
shortstay = 168
longstay = 504

# define exceptions
exceptions = ['los_hours', 'stay_class', 'los_days']

# define colours
default_palette = sns.color_palette()
colors = {0: default_palette[1], 1: default_palette[0], 2: default_palette[2]}

####################################################################################################
#                                                                                                  #
# Step 1:                                                                                          #
# Functions library                                                                                #
#                                                                                                  #
# These are the functions used in the script, and the custom callbacks.                            #
#                                                                                                  #
####################################################################################################


# function for logging output of console.
def log_output(output, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(output + '\n')
    print(output)

# callback for stopping training after a time limit has been reached
class TimeLimitCallback(Callback):
    def __init__(self, max_duration_seconds):
        self.start_time = None
        self.max_duration_seconds = max_duration_seconds

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > self.max_duration_seconds:
            self.model.stop_training = True
            print(f'Training stopped after reaching the time limit of {self.max_duration_seconds} seconds.')

# create and compile the model
def create_model(input_dim, learning_rate=learningrate, l1_reg=0.001, l2_reg=0.001):
    model = Sequential()

    # input layer
    model.add(Input(shape=(input_dim,)))

    # layer 1
    model.add(Dense(int(max_layersize), activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 2
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 3
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 4
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    # no dropout layer before passing to the output.

    # output layer, with 3 neurons for the 3 classes we are trying to predict
    model.add(Dense(3, activation='softmax'))

    # compile the model with the loss function and additional metrics
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# loadcsv2
def load_csv2(file_name, subdirectory='output_files'):
    file_path = os.path.join(target_directory, subdirectory, file_name)
    print(f'Loading {file_path}')
    return pd.read_csv(file_path, low_memory=False)

# function to create the stay classes, in case they haven't been added before.
def classify_stay(los_hours):
    if los_hours <= shortstay:
        return 0
    elif los_hours <= longstay:
        return 1
    else:
        return 2

# function to determine which class is closest for the predictor results
def closest_class(mode, mean, median):
    classes = np.array([0, 1, 2])
    combined = np.array([mode, mean, median])
    closest = np.round(np.mean(combined)).astype(int)
    return np.clip(closest, classes.min(), classes.max())


####################################################################################################
#                                                                                                  #
# Step 1:                                                                                          #
# Preprocess data and start script                                                                 #
#                                                                                                  #
# Adds the stay class, and performs the pre-processing, as well as splitting the data into 8 sets  #
# and shuffling them so that we can do cross validation.                                           #
#                                                                                                  #
####################################################################################################


# start the script timer
start_time = time.time()
print(f'the time is:{time.strftime('%m%d-%H%M')}')
print(f'at script begin.')

# get the file name for output by adding classifier for classification model and the current time
current_time = f'classifier-{time.strftime('%m%d-%H%M')}'

target_file = f'901_cleaned_combined_data{target_hours}_{target_entries}_{target_catlimit}.csv'
target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\output_files'
core_test_file_path = os.path.join(target_directory, target_file)
log_dir = os.path.dirname(core_test_file_path)
output_path = os.path.join(log_dir, f'model_output_{current_time}.txt')

plot_dir = create_directory('plots-classifier')
plot_dir = os.path.join(target_directory, plot_dir)

# load the input data
log_output('Loading core data', output_path)
core_data = pd.read_csv(core_test_file_path, low_memory=False)

# process the stay class data
core_data['stay_class'] = core_data['los_hours'].apply(classify_stay)

# split the data, to ensure that each of these 8 groups have an equal spread of stay classes in them
core_data['class_index'] = -1
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=random_state)

for i, (_, test_index) in enumerate(skf.split(core_data, core_data['stay_class'])):
    core_data.loc[test_index, 'class_index'] = i + 1

# creating the pairs for the test/train sets keeping them to mostly unique pairs. this is a form of cross validation,
# and also ensures that the final dataset is not influenced by other runs of the model
test_set_pairs_1 = [(1, 2), (3, 4), (5, 6), (7, 8)]
test_set_pairs_2 = [(1, 3), (2, 5), (4, 7), (6, 8)]
test_set_pairs_3 = [(1, 4), (2, 6), (3, 8), (5, 7)]
test_set_pairs_4 = [(1, 5), (2, 7), (3, 6), (4, 8)]
test_set_pairs_5 = [(1, 6), (2, 8), (3, 7), (4, 5)]
test_set_pairs_6 = [(1, 7), (2, 4), (3, 5), (6, 8)]

all_test_set_pairs = [test_set_pairs_1 , test_set_pairs_3, test_set_pairs_5]

#all_test_set_pairs = [test_set_pairs_1, test_set_pairs_2, test_set_pairs_3,
#                     test_set_pairs_4, test_set_pairs_5, test_set_pairs_6]

# prepare dataframe to store predictions
pred_values = core_data[['HADM_ID', 'los_hours', 'stay_class', 'los_days']].copy()

# defining the pre-process pipeline
numeric_features = core_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in exceptions + ['HADM_ID', 'class_index']]

categorical_features = core_data.select_dtypes(include=[object]).columns.tolist()
categorical_features = [col for col in categorical_features if col not in ['HADM_ID', 'class_index', 'stay_class']]

if num_only == 1:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
else:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

####################################################################################################
#                                                                                                  #
# Step 2:                                                                                          #
# Running the model                                                                                #
#                                                                                                  #
# Simply runs the model for any test set pairs, and saves the prediction data for later.           #
#                                                                                                  #
####################################################################################################


# finding the number of model runs, and allocating them each equal time
num_model_runs = len(all_test_set_pairs) * len(all_test_set_pairs[0])
time_limit_per_run = total_time_limit / num_model_runs

# looping through the sets
for set_num, test_set_pairs in enumerate(all_test_set_pairs, 1):
    log_output(f'Processing test set pairs set {set_num}', output_path)

    combined_test_preds = np.zeros(len(core_data))

    # looping through the pairs within the sets
    for pair_num, (test_set_1, test_set_2) in enumerate(test_set_pairs, 1):
        log_output(f'Processing pair {pair_num} in set {set_num}: test sets {test_set_1} and {test_set_2}',
                   output_path)

        train_data = core_data[~core_data['class_index'].isin([test_set_1, test_set_2])]
        test_data = core_data[core_data['class_index'].isin([test_set_1, test_set_2])]

        X_train = train_data.drop(columns=exceptions + ['HADM_ID', 'class_index'])
        y_train = train_data['stay_class']
        X_test = test_data.drop(columns=exceptions + ['HADM_ID', 'class_index'])
        y_test = test_data['stay_class']

        # preprocess
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # train model
        model = create_model(X_train_preprocessed.shape[1], learning_rate=learningrate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=earlystopping_patience,
                                       verbose=1, restore_best_weights=True,
                                       start_from_epoch=earlystopping_start)

        time_limit_callback = TimeLimitCallback(max_duration_seconds=time_limit_per_run)
        history = model.fit(X_train_preprocessed, y_train, epochs=epochs, batch_size=batchsize, verbose=1,
                            validation_split=testsize, callbacks=[early_stopping, time_limit_callback])

        # predict the test data
        test_preds = model.predict(X_test_preprocessed)
        test_preds = np.argmax(test_preds, axis=1)  # this takes probabilities and turns it into the class

        # add the predictions to our prediction array
        combined_test_preds[test_data.index] = test_preds

    # add the array into the df, for future use
    pred_col_name = f'pred_value_set_{set_num}'
    pred_values[pred_col_name] = combined_test_preds


####################################################################################################
#                                                                                                  #
# Step 3:                                                                                          #
# Making the file                                                                                  #
#                                                                                                  #
# This is the part where we take the predicted classes, do a small bit of feature engineering      #
# which I hope is useful, and save it to file, so that I can feed it into the regressor.           #
#                                                                                                  #
####################################################################################################

# sort the prediction values by HADM_ID to undo any shuffling
pred_values = pred_values.sort_values(by='HADM_ID')

# combining the pred_values into the main data
df = pred_values

# keep the HADM_ID, stay_class, and los_hours columns
df_transformed = df.copy()

# calculate mean, median, and mode pred class for each row
pred_class_cols = [col for col in df.columns if col.startswith('pred_value_set_')]
df_transformed['mean_pred_class'] = df[pred_class_cols].mean(axis=1)
df_transformed['median_pred_class'] = df[pred_class_cols].median(axis=1)

# calculate mode pred_class
try:
    df_transformed['mode_pred_class'] = df[pred_class_cols].mode(axis=1).iloc[:, 0]
except IndexError:
    df_transformed['mode_pred_class'] = np.nan

# calculate mean pred_value for all sets for each class
for i in range(3):
    pred_value_cols = [col for col in df.columns if col.startswith(f'pred_value_set_')]
    df_transformed[f'mean_pred_value_class_{i}'] = df_transformed[pred_value_cols].mean(axis=1)

# calculate cumulative pred_value for all sets for each class
for i in range(3):
    pred_value_cols = [col for col in df.columns if col.startswith(f'pred_value_set_')]
    df_transformed[f'cumulative_pred_value_class_{i}'] = df_transformed[pred_value_cols].sum(axis=1)

# create the pred_stay_class column
df_transformed['pred_stay_class'] = df_transformed.apply(
    lambda row: closest_class(row['mode_pred_class'], row['mean_pred_class'], row['median_pred_class']),
    axis=1
)
stay_class_idx = df_transformed.columns.get_loc('stay_class')
df_transformed.insert(stay_class_idx + 1, 'pred_stay_class', df_transformed.pop('pred_stay_class'))

print(df_transformed.head(5))

# load the original DataFrame to merge with
original_df = pd.read_csv(core_test_file_path)

# merge the aggregated prediction dataframe with the original
merged_df = pd.merge(original_df, df_transformed.drop(columns=['los_hours', 'los_days']),
                     on=['HADM_ID', 'stay_class'], how='outer')

columns_order = ['HADM_ID', 'stay_class', 'pred_stay_class'] + \
                [col for col in merged_df.columns if col not in ['HADM_ID', 'stay_class', 'pred_stay_class']]
merged_df = merged_df[columns_order]
# ok this is a bit messy but it works

####################################################################################################
#                                                                                                  #
# Step 4:                                                                                          #
# Feature importance section                                                                       #
#                                                                                                  #
# This is the part where we determine feature importance using a Gradient Boosted Classifier.      #
# Is it useful? Yes it is.                                                                         #
#                                                                                                  #
####################################################################################################


# identify the worst prediction
errors = np.abs(y_test - test_preds)
max_error_index = np.argmax(errors)
worst_prediction_actual = y_test.iloc[max_error_index]
worst_prediction_predicted = test_preds[max_error_index]

log_output(f'worst prediction - actual value: {worst_prediction_actual}, predicted value: '
           f'{worst_prediction_predicted}', output_path)

# find the features of the worst prediction
worst_prediction_features = X_test.iloc[max_error_index]
log_output(f'features of the worst prediction: {worst_prediction_features.to_dict()}', output_path)

if calc_feature_import == 1:
    # feature importance (gradient boosting classifier)
    log_output('computing feature importance using Gradient Boosting Classifier', output_path)

    try:
        gbr = GradientBoostingClassifier()
        gbr.fit(X_train_preprocessed, y_train)

        importances = gbr.feature_importances_
        indices = np.argsort(importances)[::-1]

        # find and log top n feature importances
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_feature_names = preprocessor.get_feature_names_out()[top_indices]
        log_output(f'Top {top_n} feature importances:', output_path)
        for i in range(top_n):
            log_output(f'{i + 1}. feature {top_feature_names[i]} ({top_importances[i]})', output_path)

        # plot top feature importances
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_importances, y=top_feature_names)
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')

            feature_importance_path = os.path.join(plot_dir, f'feature_importance_{current_time}.png')
            plt.savefig(feature_importance_path)
            log_output(f'feature importance plot saved to {feature_importance_path}', output_path)

            plt.show()
            plt.close()
        except Exception as e:
            log_output(f'error plotting feature importance: {e}', output_path)

    except Exception as e:
        log_output(f'Error during feature importance calculation: {e}', output_path)
else:
    log_output('Feature importance calculation is skipped.', output_path)


####################################################################################################
#                                                                                                  #
# Step 5:                                                                                          #
# Plotting                                                                                         #
#                                                                                                  #
# Simply make some plots. I got a bit experimental here; some of these charts are borderline       #
# useless, but it's nice to visualize.                                                             #
#                                                                                                  #
####################################################################################################


# plot training & validation loss values
try:
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'model loss for {target_hours} hours after admission and {target_entries} entries')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'validation loss'], loc='upper right')

    loss_plot_path = os.path.join(plot_dir, f'model_loss_{current_time}.png')
    create_directory(os.path.dirname(loss_plot_path))
    plt.savefig(loss_plot_path)
    log_output(f'loss plot saved to {loss_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting loss: {e}', output_path)

# confusion matrix (absolute values)
try:
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Absolute Values) for {target_hours} hours after admission and {target_entries} entries')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_plot_path = os.path.join(plot_dir, f'confusion_matrix_abs_{current_time}.png')
    create_directory(os.path.dirname(cm_plot_path))
    plt.savefig(cm_plot_path)
    log_output(f'confusion matrix (absolute values) saved to {cm_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting confusion matrix (absolute values): {e}', output_path)

# confusion matrix (% values)
try:
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'Confusion Matrix (Percentage Values) for {target_hours} hours after admission and {target_entries} entries')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_percentage_plot_path = os.path.join(plot_dir, f'confusion_matrix_percentage_{current_time}.png')
    create_directory(os.path.dirname(cm_percentage_plot_path))
    plt.savefig(cm_percentage_plot_path)
    log_output(f'confusion matrix (percentage values) saved to {cm_percentage_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting confusion matrix (percentage values): {e}', output_path)


# ROC new and improved, hopefully for all 3 classes now.

# binarize the output labels
y_pred_prob = model.predict(X_test_preprocessed)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Assuming 3 classes (0, 1, 2)
n_classes = y_test_bin.shape[1]

#  ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# plot all ROC curves
plt.figure(figsize=(10, 7))
for i in range(n_classes):
    color = colors[i]
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve per class')
plt.legend(loc="lower right")

# save
roc_curve_path = os.path.join(plot_dir, f'roc_curve_{current_time}.png')
plt.savefig(roc_curve_path)
log_output(f'ROC curve saved to {roc_curve_path}', output_path)

plt.show()
plt.close()

# class distribution
try:
    plt.figure(figsize=(10, 7))
    true_classes = pd.Series(y_test, name='True Class')
    pred_classes = pd.Series(test_preds, name='Predicted Class')
    df = pd.concat([true_classes, pred_classes], axis=1)
    df_melt = df.melt(var_name='Type', value_name='Class')
    sns.countplot(x='Class', hue='Type', data=df_melt)
    plt.title(f'Class Distribution: True vs Predicted for {target_hours} hours after admission and {target_entries} entries')
    plt.xlabel('Class')
    plt.ylabel('Count')

    class_dist_path = os.path.join(plot_dir, f'class_distribution_{current_time}.png')
    plt.savefig(class_dist_path)
    log_output(f'class distribution plot saved to {class_dist_path}', output_path)

    plt.show()
    plt.close()
except Exception as e:
    log_output(f'error plotting class distribution: {e}', output_path)


####################################################################################################
#                                                                                                  #
# Step 6:                                                                                          #
# Save the combined output file                                                                    #
#                                                                                                  #
####################################################################################################


# create the final output file name
final_output_file = f'902_full_data_pred_{target_hours}_{target_entries}.csv'

# save the final merged DataFrame
merged_df.to_csv(os.path.join(target_directory, final_output_file), index=False)
print(f'Final merged DataFrame saved to {final_output_file}')

####################################################################################################
#                                                                                                  #
# Step 7:                                                                                          #
# Summary of results                                                                               #
#                                                                                                  #
####################################################################################################


# generate and save a summary table
summary_file_path = os.path.join(log_dir, f'summary_{current_time}.txt')
with open(summary_file_path, 'w', encoding='utf-8') as f:
    f.write(f'Target file name: {target_file}\n')
    f.write(f'Hyperparameters:\n')
    f.write(f' - epochs: {epochs}\n')
    f.write(f' - learning rate: {learningrate}\n')
    f.write(f' - batch size: {batchsize}\n')
    f.write(f' - test size: {testsize}\n')
    f.write(f' - early stopping start: {earlystopping_start}\n')
    f.write(f' - early stopping patience: {earlystopping_patience}\n')
    f.write(f' - total time limit: {total_time_limit}\n')
    f.write(f' - max layer size: {max_layersize}\n')
    f.write(f' - l1 regularization: {l1reg}\n')
    f.write(f' - l2 regularization: {l2reg}\n')

    f.write(f'Results:\n')
    f.write(f' - Overall Accuracy: {accuracy_score(y_test, test_preds):.4f}\n')

    # write roc auc for each class
    for cls in range(n_classes):
        f.write(f' - Roc Auc for class {cls}: {roc_auc[cls]:.4f}\n')

    # precision for each class
    precision_report = classification_report(y_test, test_preds, output_dict=True)
    for cls in range(n_classes):
        f.write(f' - Precision for class {cls}: {precision_report[str(cls)]["precision"]:.4f}\n')

    # overall precision
    f.write(f' - Overall Precision: {precision_report["weighted avg"]["precision"]:.4f}\n')

log_output(f'Summary saved to {summary_file_path}', output_path)


# calculate and print the execution time
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
log_output(f'Script took {elapsed_time:.2f} seconds to execute.', output_path)
log_output(f'Thats {int(hours)}:{int(minutes)}:{int(seconds)} (hh:mm:ss).', output_path)
