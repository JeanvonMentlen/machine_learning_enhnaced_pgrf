import random
import numpy as np
from data_pipeline import load_data, load_labels
# Adjust these paths and filenames as needed
nr = 13

# Split proportions (these should add up to 1)
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

##################################################################################################################
# Data
##################################################################################################################
iteration = 4
type = 'data'
input_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/{nr}param_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
train_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/data/train/{nr}param_train_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
validation_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/data/validation/{nr}param_validation_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
test_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/data/test/{nr}param_test_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
data_file = np.loadtxt(input_file, delimiter='\t', skiprows=1)

q_vector = data_file[:, 0]
data = data_file[:, 1:]
nan_columns = np.all(np.isnan(data), axis=0)
data = data[:, ~nan_columns]


# Calculate the split indices 0 for labels, 1 for data
train_idx = int(train_ratio * data.shape[1])
validation_idx = int((train_ratio + validation_ratio) * data.shape[1])

# Split the data switch columns and rows for labels and data
train_data = data[:, :train_idx]
validation_data = data[:, train_idx:validation_idx]
test_data = data[:, validation_idx:]

# Add the q_vector to the beginning of each data split
train_data = np.column_stack((q_vector, train_data))
validation_data = np.column_stack((q_vector, validation_data))
test_data = np.column_stack((q_vector, test_data))

np.savetxt(train_file, train_data, delimiter='\t')
np.savetxt(validation_file, validation_data, delimiter='\t')
np.savetxt(test_file, test_data, delimiter='\t')

##################################################################################################################
# Labels
##################################################################################################################

type = 'labels'
input_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/{nr}param_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
# input_file2 = 'full_{nr}param_{type}.txt'.format(nr=nr, type=type)
train_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/data/train/{nr}param_train_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
validation_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting//data/validation/{nr}param_validation_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
test_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/data/test/{nr}param_test_{type}_{it}.txt'.format(nr=nr, type=type, it=iteration)
label_file = '/local/home/sujataj/PythonProjects/sas-ml-fitting/7param_labels_stats.txt'

labels = np.loadtxt(input_file, delimiter='\t', skiprows=1)
labels = labels[~nan_columns, :]

# Calculate the split indices 0 for labels, 1 for data
train_idx = int(train_ratio * labels.shape[0])
validation_idx = int((train_ratio + validation_ratio) * labels.shape[0])

# Split the data switch columns and rows for labels and data
train_labels = labels[:train_idx, :]
validation_labels = labels[train_idx:validation_idx, :]
test_labels = labels[validation_idx:, :]


# Save the splits to separate files
np.savetxt(train_file, train_labels, delimiter='\t')
np.savetxt(validation_file, validation_labels, delimiter='\t')
np.savetxt(test_file, test_labels, delimiter='\t')
