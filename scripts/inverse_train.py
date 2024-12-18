import os
import datetime
import numpy as np
import torch.optim as optim
from custom_dataset import CustomDataset
from models import InverseCNN
import config_reader
from data_pipeline import data_preprocessing, noise_free, experimental_bg
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR
from performance_plotting import plot_training_performance, plot_forward_prediction_performance, \
    plot_forward_prediction_excerpt, plot_inverse_prediction_performance
from train import train, test_model
import sys

def main():
    now = datetime.datetime.now()
    logdir = 'trained_models/' + now.strftime('%Y-%m-%d-%H%M')
    logdir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

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

    training_dataset = CustomDataset(torch.tensor(training_intensities, dtype=torch.float32),
                                     torch.tensor(training_parameters, dtype=torch.float32))

    validation_dataset = CustomDataset(torch.tensor(validation_intensities, dtype=torch.float32),
                                       torch.tensor(validation_parameters, dtype=torch.float32))

    test_dataset = CustomDataset(torch.tensor(test_intensities, dtype=torch.float32),
                                 torch.tensor(test_parameters, dtype=torch.float32))

    # Create a DataLoader with the desired batch size
    train_dataloader = DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Forward Model: Parameters as input, intensity curve as prediction
    # ------------------------------------------------------------------------------------------------------------------

    number_of_q_values = len(q_vector)
    number_of_parameters = training_parameters.shape[1]

    inverse_model = InverseCNN(number_of_q_values, number_of_parameters)
    num_params = sum(p.numel() for p in inverse_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    loss_function = torch.nn.MSELoss()  # You can replace this with another loss function if needed
    # loss_function = torch.nn.L1Loss()  # You can replace this with another loss function if needed
    optimizer = optim.Adam(inverse_model.parameters(), lr=8e-4)  # Replace 0.001 with your desired learning rate
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    # ------------------------------------------------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------------------------------------------------

    training_history, validation_history = train(inverse_model, train_dataloader, validation_dataloader, NUM_EPOCHS,
                                                 optimizer, loss_function, scheduler, logdir, model_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate and print metrics, validate, and/or save the model, as needed
    # ------------------------------------------------------------------------------------------------------------------

    test_output, test_target = test_model(inverse_model, test_dataloader)

    np.savetxt(logdir + '/test_output.npy', test_output, delimiter='\t')
    np.savetxt(logdir + '/test_target.npy', test_target, delimiter='\t')

    plot_training_performance(training_history, validation_history, NUM_EPOCHS, logdir,
                              '/{}param_forward_loss_evolution'.format(number_of_parameters))

    plot_inverse_prediction_performance(test_output, test_target, logdir,
                                    '/{}param_forward_prediction_performance'.format(number_of_parameters))

    print('Done')




if __name__ == '__main__':
    main()
