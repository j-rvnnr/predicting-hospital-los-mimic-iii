import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model

# these are hyperparameters for the neural network
max_layersize = 512
l1reg = 0.001
l2reg = 0.001
dropout = 0.5

# specify the target directory where model visualizations will be saved
target_directory = r'C:\Users\ander\Documents\.Uni\Project\mimic-iii-clinical-database-1.4\.unpacked\output_files'

# make sure the directory exists
os.makedirs(target_directory, exist_ok=True)

# classifier model
def create_classifier_model(input_dim):
    model = Sequential()

    # input layer
    model.add(Input(shape=(input_dim,)))

    # layer 1
    model.add(Dense(int(max_layersize), activation='relu',
                    kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 2
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 3
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 4
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())
    # no dropout layer before passing to the output.

    # output layer, with 3 neurons for the 3 classes we are trying to predict
    model.add(Dense(3, activation='softmax'))

    return model

# regressor model
def create_regressor_model(input_dim):
    model = Sequential()

    # input layer
    model.add(Input(shape=(input_dim,)))

    # layer 1. contains dense, regularisation, batch normalisation, relu (random or parametric) and dropout
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 2, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 3, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # layer 4, dense, regularisation, relu and dropout
    model.add(Dense(int(max_layersize), activation='relu', kernel_regularizer=l1_l2(l1=l1reg, l2=l2reg)))
    model.add(PReLU())

    # output layer, with a single node and linear activation
    model.add(Dense(1, activation='linear'))

    return model

# create the classifier and regressor models without running them
classifier_model = create_classifier_model(input_dim=20)
regressor_model = create_regressor_model(input_dim=20)

# save the model visualizations
plot_model(classifier_model, to_file=os.path.join(target_directory, 'classifier_model.png'), show_shapes=True, show_layer_names=True)
plot_model(regressor_model, to_file=os.path.join(target_directory, 'regressor_model.png'), show_shapes=True, show_layer_names=True)
