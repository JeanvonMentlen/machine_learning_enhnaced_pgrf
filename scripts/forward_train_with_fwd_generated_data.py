import os
import datetime
import numpy as np
import torch.optim as optim
from custom_dataset import CustomDataset
from models import ForwardCNN, EnsembleModel
import config_reader
from data_pipeline import data_preprocessing, noise_free, experimental_bg
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR
from performance_plotting import plot_training_performance, plot_forward_prediction_performance, \
    plot_forward_prediction_excerpt, plot_inverse_prediction_performance
from train import train, test_model
from numpy.random import uniform
import pandas as pd


def main():
    now = datetime.datetime.now()
    logdir = './trained_models/' + now.strftime('%Y-%m-%d-%H%M')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config = config_reader.ConfigLoader('../configs/params.config')
    model_name = config.get_model_name()
    NUM_EPOCHS = config.get_number_of_epochs()

    TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, TEST_BATCH_SIZE = config.get_batch_sizes()

    # ------------------------------------------------------------------------------------------------------------------
    # Load Training and Validation Data first
    # ------------------------------------------------------------------------------------------------------------------
    n_train = 250000
    n_validate = 60000
    n_test = 60000

    int_const = uniform(0, 1, n_train + n_validate + n_test)
    phi_A = uniform(0, 1, n_train + n_validate + n_test)  # 0.89
    vol_factor = uniform(0.32, 0.32, n_train + n_validate + n_test)  # 0.31
    beta_angle = uniform(0.92, 0.92, n_train + n_validate + n_test)  # 0.91
    l_z = uniform(0, 1, n_train + n_validate + n_test)
    l_y = uniform(0, 1, n_train + n_validate + n_test)
    d_z = uniform(0.8, 0.8, n_train + n_validate + n_test)  # 0.7
    # porod_const = uniform(0, 1, n_train + n_validate + n_test)

    parameters = np.vstack((int_const, phi_A, vol_factor, beta_angle, l_z, l_y, d_z))
    parameters = parameters.T

    training_parameters = parameters[:n_train, :]
    validation_parameters = parameters[n_train + 1:n_train + 1 + n_validate, :]
    test_parameters = parameters[n_validate + 1:n_validate + 1 + n_test, :]

    model_paths = ["../trained_models/forwardCNN_7param_6/SANS_CNN.pth",
                "../trained_models/forwardCNN_7param_7/SANS_CNN.pth",
                   "../trained_models/forwardCNN_7param_8/SANS_CNN.pth",
                   "../trained_models/forwardCNN_7param_9/SANS_CNN.pth"
                   ]

    models = [torch.load(model_path) for model_path in model_paths]

    forward_ensemble_CNN = EnsembleModel(models)
    forward_ensemble_CNN.eval()
    forward_ensemble_CNN = forward_ensemble_CNN.double()

    training_intensities = list()
    for training_parameters_batch in np.split(training_parameters, n_train / 100):
        train_param_batch = torch.from_numpy(training_parameters_batch).unsqueeze(2)
        train_param_batch = train_param_batch.double()
        training_intensities_batch = forward_ensemble_CNN(train_param_batch)
        training_intensities.append(training_intensities_batch.detach().numpy())

    training_intensities = np.concatenate(training_intensities, axis=0)

    validation_intensities = list()
    for validation_parameters_batch in np.split(validation_parameters, n_validate / 100):
        val_param_batch = torch.from_numpy(validation_parameters_batch).unsqueeze(2)
        val_param_batch = val_param_batch.double()
        validation_intensities_batch = forward_ensemble_CNN(val_param_batch)
        validation_intensities.append(validation_intensities_batch.detach().numpy())

    validation_intensities = np.concatenate(validation_intensities, axis=0)

    test_intensities = list()
    for test_parameters_batch in np.split(test_parameters, n_test / 100):
        test_param_batch = torch.from_numpy(test_parameters_batch).unsqueeze(2)
        test_param_batch = test_param_batch.double()
        test_intensities_batch = forward_ensemble_CNN(test_param_batch)
        test_intensities.append(test_intensities_batch.detach().numpy())

    test_intensities = np.concatenate(test_intensities, axis=0)

    (training_data_file_name, training_labels_file_name, validation_data_file_name,
     validation_labels_file_name, test_data_file_name, test_labels_file_name) = config.get_file_names()

    q_vector, _, _, _, _ = data_preprocessing(
        data_file_name=training_data_file_name,
        labels_file_name=training_labels_file_name,
        q_max=117,
        noise_fct=experimental_bg)

    # ------------------------------------------------------------------------------------------------------------------
    # Create your custom dataset (assuming data and labels are numpy arrays)
    # ------------------------------------------------------------------------------------------------------------------
    cols = [0, 1, 4, 5]
    training_dataset = CustomDataset(torch.tensor(training_parameters[:, cols], dtype=torch.float32),
                                     torch.tensor(training_intensities, dtype=torch.float32))

    validation_dataset = CustomDataset(torch.tensor(validation_parameters[:, cols], dtype=torch.float32),
                                       torch.tensor(validation_intensities, dtype=torch.float32))

    test_dataset = CustomDataset(torch.tensor(test_parameters[:, cols], dtype=torch.float32),
                                 torch.tensor(test_intensities, dtype=torch.float32))

    # Create a DataLoader with the desired batch size
    train_dataloader = DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Forward Model: Parameters as input, intensity curve as prediction
    # ------------------------------------------------------------------------------------------------------------------

    number_of_q_values = len(q_vector)
    number_of_parameters = training_parameters[:, cols].shape[1]

    forward_model = ForwardCNN(number_of_parameters, number_of_q_values)
    num_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    loss_function = torch.nn.MSELoss()  # You can replace this with another loss function if needed
    # loss_function = torch.nn.L1Loss()  # You can replace this with another loss function if needed
    optimizer = optim.Adam(forward_model.parameters(), lr=8e-4)  # Replace 0.001 with your desired learning rate
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    # ------------------------------------------------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------------------------------------------------

    training_history, validation_history = train(forward_model, train_dataloader, validation_dataloader, NUM_EPOCHS,
                                                 optimizer, loss_function, scheduler, logdir, model_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate and print metrics, validate, and/or save the model, as needed
    # ------------------------------------------------------------------------------------------------------------------

    test_output, test_target = test_model(forward_model, test_dataloader)

    np.savetxt(logdir + '/test_output.npy', test_output, delimiter='\t')
    np.savetxt(logdir + '/test_target.npy', test_target, delimiter='\t')

    train_history_df = pd.DataFrame(training_history, columns=['mse', 'accuracy'])
    train_history_df.to_csv(logdir + '/training_history.csv', sep='\t')
    validation_history_df = pd.DataFrame(validation_history, columns=['mse', 'accuracy'])
    validation_history_df.to_csv(logdir + '/validation_history.csv', sep='\t')

    plot_training_performance(training_history, validation_history, NUM_EPOCHS, logdir,
                              '/{}param_forward_loss_evolution'.format(number_of_parameters))
    plot_forward_prediction_performance(test_output, test_target, q_vector, logdir,
                                        '/{}param_forward_prediction_performance'.format(number_of_parameters))


if __name__ == '__main__':
    main()
