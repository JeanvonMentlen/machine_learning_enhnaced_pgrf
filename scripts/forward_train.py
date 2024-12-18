import os
import datetime
import numpy as np
import torch.optim as optim
from custom_dataset import CustomDataset
from models import ForwardCNN
import config_reader
from data_pipeline import data_preprocessing, noise_free, experimental_bg
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from performance_plotting import plot_training_performance, plot_forward_prediction_performance, \
    plot_forward_prediction_excerpt
from train import train, test_model
import sys


def main():
    now = datetime.datetime.now()
    logdir = '../trained_models/' + now.strftime('%Y-%m-%d-%H%M')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # change parameters in params.config file
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', 'configs/params.config'))
    config = config_reader.ConfigLoader(config_path)
    model_name = config.get_model_name()
    NUM_EPOCHS = config.get_number_of_epochs()

    TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, TEST_BATCH_SIZE = config.get_batch_sizes()

    # ------------------------------------------------------------------------------------------------------------------
    # Load Training and Validation Data first
    # ------------------------------------------------------------------------------------------------------------------

    (training_data_file_name, training_labels_file_name, validation_data_file_name,
     validation_labels_file_name, test_data_file_name, test_labels_file_name) = config.get_file_names()

    # label statistics are irrelevant for the training, hence the _
    q_vector, training_intensities, training_parameters, _, _ = data_preprocessing(
        data_file_name=training_data_file_name,
        labels_file_name=training_labels_file_name,
        q_max=117,
        noise_fct=experimental_bg)

    _, validation_intensities, validation_parameters, _, _ = data_preprocessing(
        data_file_name=validation_data_file_name,
        labels_file_name=validation_labels_file_name,
        q_max=117,
        noise_fct=experimental_bg)

    _, test_intensities, test_parameters, _, _ = data_preprocessing(
        data_file_name=test_data_file_name,
        labels_file_name=test_labels_file_name,
        q_max=117,
        noise_fct=experimental_bg)

    # ------------------------------------------------------------------------------------------------------------------
    # Create your custom dataset (assuming data and labels are numpy arrays)
    # ------------------------------------------------------------------------------------------------------------------

    training_dataset = CustomDataset(torch.tensor(training_parameters, dtype=torch.float32),
                                     torch.tensor(training_intensities, dtype=torch.float32))

    validation_dataset = CustomDataset(torch.tensor(validation_parameters, dtype=torch.float32),
                                       torch.tensor(validation_intensities, dtype=torch.float32))

    test_dataset = CustomDataset(torch.tensor(test_parameters, dtype=torch.float32),
                                 torch.tensor(test_intensities, dtype=torch.float32))

    # Create a DataLoader with the desired batch size
    train_dataloader = DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Forward Model: Parameters as input, intensity curve as prediction
    # ------------------------------------------------------------------------------------------------------------------

    number_of_q_values = len(q_vector)
    number_of_parameters = training_parameters.shape[1]

    forward_model = ForwardCNN(number_of_parameters, number_of_q_values)
    num_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # loss_function = torch.nn.MSELoss()  # You can replace this with another loss function if needed
    loss_function = torch.nn.L1Loss()  # You can replace this with another loss function if needed
    optimizer = optim.Adam(forward_model.parameters(),
                           lr=8e-4)#, weight_decay=1e-7)  # L2 normalization; not necessary
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

    np.savetxt(logdir + '/test_output.txt', test_output, delimiter='\t')
    np.savetxt(logdir + '/test_target.txt', test_target, delimiter='\t')

    train_history_df = pd.DataFrame(training_history, columns=['mse', 'accuracy'])
    train_history_df.to_csv(logdir + '/training_history.csv', sep='\t')
    validation_history_df = pd.DataFrame(validation_history, columns=['mse', 'accuracy'])
    validation_history_df.to_csv(logdir + '/validation_history.csv', sep='\t')

    plot_training_performance(training_history, validation_history, NUM_EPOCHS, logdir,
                              '/{}param_forward_loss_evolution'.format(number_of_parameters))
    plot_forward_prediction_performance(test_output, test_target, q_vector, logdir,
                                        '/{}param_forward_prediction_performance'.format(number_of_parameters))
    plot_forward_prediction_excerpt(test_output, test_target, q_vector, logdir,
                                    '/{}param_forward_prediction_excerpt'.format(number_of_parameters))


if __name__ == '__main__':
    main()
