import gc
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU, Layer
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf



####################################################################################################
#                                                                                                  #
# step 0:                                                                                          #
# Initiatialisation.                                                                               #
#                                                                                                  #
# here we set our variables which we will use during the course of the script. They are broken     #
# down into sections, which should be mostly chronological in the order in which they appear in    #
# the script, although they may not be 100%. They are also grouped roughly by which function they  #
# are used in.                                                                                     #
#                                                                                                  #
####################################################################################################


# these variables are for data handling, whether we wish to include the longstay, only look at the longstay, and
# whether we want to include categorical data.
exclude_longstay = 1
only_longstay = 0
num_only = 1

# these are hyperparameters for the neural network, including runtime, learningrate, batch size and the train/test
# split size.
epochs = 400
learningrate = 0.0005
batchsize = 32
testsize = 0.2

# earlystopping parameters
earlystopping_start = 200
earlystopping_patience = 50

# total time limit (seconds)[36000 is 10 hours]
total_time_limit = 36000

# neural network parameters
max_layersize = 512
l1reg = 0.001
l2reg = 0.001
dropout = 0.4

# miscellaneous other parameters for the model, the data and feature importance limiter
xval_kfolds = 2
random_state = 117
top_n = 50


####################################################################################################
#                                                                                                  #
# step 1:                                                                                          #
# Defining our functions.                                                                          #
#                                                                                                  #
# This section is where we define the functions which we will use in the model, including the      #
# console logger, the custom random relu and huber loss functions, and the model itself.           #
#                                                                                                  #
####################################################################################################


# function for logging output of console.
def log_output(output, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(output + '\n')
    print(output)

# random relu custom layer
class RandomReLU(Layer):
    def __init__(self, lower=0.3, upper=0.6, **kwargs):
        super(RandomReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=None):
        if training:
            alpha = K.random_uniform(shape=[], minval=self.lower, maxval=self.upper)
        else:
            alpha = (self.lower + self.upper) / 2
        return K.maximum(alpha * inputs, inputs)

# huber loss custom metric
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = tf.square(error) / 2
    large_error_loss = delta * (tf.abs(error) - delta / 2)
    return tf.where(is_small_error, small_error_loss, large_error_loss)

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
            print(f"Training stopped after reaching the time limit of {self.max_duration_seconds} seconds.")


# compile the model in this function
def create_model(input_dim, learning_rate=learningrate, l1_reg=0.001, l2_reg=0.001):
    model = Sequential()

    # input layer
    model.add(Input(shape=(input_dim,)))

    # layer 1. contains dense, regularisation, batch normalisation, relu (random or parametric) and dropout
    model.add(Dense(int(max_layersize), input_dim=input_dim, activation='relu',
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 2, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize/2), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 3, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize/4), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 4, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize/8), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 5, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize/16), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # output layer, with a single node and linear activation
    model.add(Dense(1, activation='linear'))

    # compile the model with our loss functions, as well as some additional metrics to examine model performance
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=huber_loss,
        metrics=[
            'mae',
            'mse',
            huber_loss
        ]
    )
    return model

# function to create a directory
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

out_dir = create_directory('output_files')


####################################################################################################
#                                                                                                  #
# step 2:                                                                                          #
# Preprocessing                                                                                    #
#                                                                                                  #
# Here we load the data, preprocess it for the model and prepare for the machine learning step.    #
#                                                                                                  #
####################################################################################################


# start the script timer
start_time = time.time()

# catch this potential error before it can cause havoc further down
if exclude_longstay == 1 and only_longstay == 1:
    only_longstay = 0
    print('exclude_longstay and only_longstay both set to 1. Setting only_longstay to 0 to prevent error')


# get the file name for output by adding regressor for regressor model and the current time
current_time = f'regressor_{time.strftime("%Y%m%d-%H%M%S")}'

# defining the file paths for the input data
target_file = '901_full_data_preddata72_5.csv'
target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\output_files'
core_test_file_path = os.path.join(target_directory, target_file)
log_dir = os.path.dirname(core_test_file_path)
output_path = os.path.join(log_dir, f'model_output_{current_time}.txt')

plot_dir = os.path.join(target_directory, 'plots-regressor')
create_directory(plot_dir)

# load the input data
log_output('loading core data', output_path)
core_data = pd.read_csv(core_test_file_path, low_memory=False)

# exclude rows where stay class = 2 if we choose to exclude long stay patients. it results in more accuracy for the
# other classes
if exclude_longstay == 1:
    core_data = core_data[core_data['stay_class'] != 2]

# if only_longstay = 1 then we filter out to be only the longstay, this is for experimental purposes, the longstay
# patients are much harder to predict
if only_longstay == 1 and exclude_longstay == 0:
    core_data = core_data[core_data['stay_class'] == 2]

# check for nan values in core_data
if core_data.isna().any().any():
    log_output('nan values found in data. Filling nans with column mean.', output_path)
    core_data.fillna(core_data.mean(), inplace=True)

# exclude los_hours and preserve hadm hours for later use (validation)
features = core_data.drop(columns=['los_hours', 'los_days_x', 'los_days_y', 'stay_class'])
target = core_data['los_hours'] # set target variable
hadm_ids = features['HADM_ID'] # save the hopsital_admission id's
features = features.drop(columns=['HADM_ID']) # load in the features we wish to examine

numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features.select_dtypes(include=[object]).columns.tolist()

# drop categorical columns if num_only = 1
if num_only == 1:
    features = features[numeric_features]
    log_output('Dropping non-numeric columns as num_only is set to 1', output_path)

# ensure our numerical columns are set to actually be known as numbers
for col in numeric_features:
    features[col] = features[col].replace({',': ''}, regex=True)  # remove commas
    features[col] = pd.to_numeric(features[col], errors='coerce')  # numeric convert, nothing = nan

# defining the pre-process pipeline
if num_only == 1:
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
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

# split the data into test and train sets, based on the testsize variable at the start
log_output('splitting data into training and test sets', output_path)
X_train, X_test, y_train, y_test, hadm_train, hadm_test = train_test_split(features, target, hadm_ids,
                                                                           test_size=testsize, random_state=random_state)
# apply the preprocessing to the data
log_output('preprocessing data', output_path)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# debug printing
log_output(f'shape of X_train: {X_train.shape}', output_path)
log_output(f'shape of X_train_preprocessed: {X_train_preprocessed.shape}', output_path)
log_output(f'shape of X_test: {X_test.shape}', output_path)
log_output(f'shape of X_test_preprocessed: {X_test_preprocessed.shape}', output_path)

# convert the data into dataframes so we can handle them a bit easirer
try:
    X_train_df = pd.DataFrame(X_train_preprocessed, columns=preprocessor.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=preprocessor.get_feature_names_out())
except Exception as e:
    log_output(f'error converting preprocessed data to dataframe: {e}', output_path)

# define early stopping, using the variables we set above
early_stopping = EarlyStopping(monitor='val_loss', patience=earlystopping_patience, verbose=1, restore_best_weights=True,
                               start_from_epoch=earlystopping_start)
# and the time limit
time_limit_callback = TimeLimitCallback(max_duration_seconds=total_time_limit)


####################################################################################################
#                                                                                                  #
# Step 3:                                                                                          #
# Training the model.                                                                              #
#                                                                                                  #
# Here we train the neural network, including the K-folds validation step, as well as averaging    #
# out and removing the extreme outliers. We also save the model at the end, so we can use it again #
# in the future.                                                                                   #
#                                                                                                  #
####################################################################################################


# train the model
kf = KFold(n_splits=xval_kfolds, shuffle=True, random_state=random_state)
fold = 1
all_test_preds = []

for train_index, val_index in kf.split(X_train_preprocessed):
    log_output(f'\nTraining fold {fold}', output_path)

    X_fold_train, X_fold_val = X_train_preprocessed[train_index], X_train_preprocessed[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    model = create_model(input_dim=X_train_preprocessed.shape[1], learning_rate=learningrate)
    history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batchsize, verbose=1,
                        validation_data=(X_fold_val, y_fold_val), callbacks=[early_stopping, time_limit_callback])

    fold_test_preds = model.predict(X_test_preprocessed)
    all_test_preds.append(fold_test_preds)

    fold += 1

# average the predictions from all folds
avg_test_preds = np.mean(all_test_preds, axis=0).flatten()

# handle extreme predictions
cap_value = np.percentile(y_test, 99)
avg_test_preds = np.where(avg_test_preds > cap_value, cap_value, avg_test_preds)

# evaluate the model on the test data
log_output('evaluating the model on the test data', output_path)
test_mse = mean_squared_error(y_test, avg_test_preds)
test_r2 = r2_score(y_test, avg_test_preds)
test_mae = mean_absolute_error(y_test, avg_test_preds)

log_output(f'test mse: {test_mse}', output_path)
log_output(f'test r2: {test_r2}', output_path)
log_output(f'test mae: {test_mae}', output_path)

# save the model
model_filename = f'model_{current_time}_r2_{test_r2:.3f}.keras'
model.save(os.path.join(log_dir, model_filename))
log_output(f'model saved to {model_filename}', output_path)


####################################################################################################
#                                                                                                  #
# Step 4:                                                                                          #
# Plotting.                                                                                        #
#                                                                                                  #
# We make some plots, of the loss values and the predicted vs actual los, and residuals. We also   #
# do the feature importance calculation in this part.                                              #
#                                                                                                  #
####################################################################################################


# plot training & validation loss values
try:
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    loss_plot_path = os.path.join(plot_dir, f'model_loss_{current_time}.png')
    plt.savefig(loss_plot_path)
    log_output(f'loss plot saved to {loss_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting loss: {e}', output_path)

# scatter plot of predicted vs actual LOS
try:
    plt.figure()
    plt.scatter(y_test, avg_test_preds, alpha=0.5)
    plt.title('predicted vs actual los')
    plt.xlabel('actual los')
    plt.ylabel('predicted los')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    scatter_plot_path = os.path.join(plot_dir, f'predicted_vs_actual_{current_time}.png')
    plt.savefig(scatter_plot_path)
    log_output(f'scatter plot saved to {scatter_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting scatter plot: {e}', output_path)

# residual Plot
try:
    residuals = y_test - avg_test_preds
    plt.figure()
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.hlines(0, min(y_test), max(y_test), colors='r', linestyles='dashed')
    plt.xlabel('Actual LOS')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    residual_plot_path = os.path.join(plot_dir, f'residual_plot_{current_time}.png')
    plt.savefig(residual_plot_path)
    log_output(f'residual plot saved to {residual_plot_path}', output_path)
except Exception as e:
    log_output(f'error plotting residuals: {e}', output_path)

# identify the worst prediction
errors = np.abs(y_test - avg_test_preds)
max_error_index = np.argmax(errors)
worst_prediction_actual = y_test.iloc[max_error_index]
worst_prediction_predicted = avg_test_preds[max_error_index]

log_output(f'worst prediction - actual value: {worst_prediction_actual}, predicted value: '
           f'{worst_prediction_predicted}',output_path)

# examine the features of the worst prediction
worst_prediction_features = X_test.iloc[max_error_index]
log_output(f'features of the worst prediction: {worst_prediction_features.to_dict()}', output_path)

# feature Importance using Gradient Boosting Regressor
log_output('computing feature importance using Gradient Boosting Regressor', output_path)
gbr = GradientBoostingRegressor()
gbr.fit(X_train_preprocessed, y_train)

importances = gbr.feature_importances_
indices = np.argsort(importances)[::-1]

# get and log top n feature importances
top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_feature_names = preprocessor.get_feature_names_out()[top_indices]
log_output(f'Top {top_n} feature importances:', output_path)
for i in range(top_n):
    log_output(f'{i + 1}. feature {top_feature_names[i]} ({top_importances[i]})', output_path)


# calculate and print the execution time
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
log_output(f"Script took {elapsed_time:.2f} seconds to execute.", output_path)
log_output(f"That's {int(hours)}:{int(minutes)}:{int(seconds)} (hh:mm:ss).", output_path)
