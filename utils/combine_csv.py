import pandas as pd
import os
import numpy as np

# Define the subfolders
subfolders = ['train', 'validation', 'test']

# Loop over the subfolders
for subfolder in subfolders:
    # Define the file paths
    data_file1 = os.path.join('../data', subfolder, '13param_{}_data_1.txt'.format(subfolder))
    data_file2 = os.path.join('../data', subfolder, '13param_{}_data_2.txt'.format(subfolder))
    data_file3 = os.path.join('../data', subfolder, '13param_{}_data_3.txt'.format(subfolder))
    data_file4 = os.path.join('../data', subfolder, '13param_{}_data_4.txt'.format(subfolder))

    label_file1 = os.path.join('../data', subfolder, '13param_{}_labels_1.txt'.format(subfolder))
    label_file2 = os.path.join('../data', subfolder, '13param_{}_labels_2.txt'.format(subfolder))
    label_file3 = os.path.join('../data', subfolder, '13param_{}_labels_3.txt'.format(subfolder))
    label_file4 = os.path.join('../data', subfolder, '13param_{}_labels_4.txt'.format(subfolder))



    data1 = np.loadtxt(data_file1, delimiter='\t')
    data2 = np.loadtxt(data_file2, delimiter='\t')
    data3 = np.loadtxt(data_file3, delimiter='\t')
    data4 = np.loadtxt(data_file4, delimiter='\t')

    label1 = np.loadtxt(label_file1, delimiter='\t')
    label2 = np.loadtxt(label_file2, delimiter='\t')
    label3 = np.loadtxt(label_file3, delimiter='\t')
    label4 = np.loadtxt(label_file4, delimiter='\t')

    # Save the combined data files
    data_combined = np.hstack((data1, data2[:, 1:], data3[:, 1:], data4[:, 1:]))

    # Combine the label files vertically
    label_combined = np.vstack((label1, label2, label3, label4))

    # Save the combined data files
    np.savetxt(os.path.join('../data', subfolder, '13param_{}_data.txt'.format(subfolder)), data_combined, delimiter='\t')
    # np.savetxt(os.path.join('../data', subfolder, '7param_{}_data_4.txt'.format(subfolder)), data_combined, delimiter='\t')

    # Save the combined label files
    np.savetxt(os.path.join('../data', subfolder, '13param_{}_labels.txt'.format(subfolder)), label_combined, delimiter='\t')
    # np.savetxt(os.path.join('../data', subfolder, '7param_{}_labels_4.txt'.format(subfolder)), label_combined, delimiter='\t')