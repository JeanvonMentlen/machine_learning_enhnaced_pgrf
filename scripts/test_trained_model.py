import os
import datetime
from custom_dataset import CustomDataset
from models import ForwardCNN, EnsembleModel
import config_reader
from data_pipeline import data_preprocessing, noise_free, experimental_bg
from torch.utils.data import DataLoader
import torch
from performance_plotting import plot_training_performance, plot_forward_prediction_performance, \
    plot_forward_prediction_excerpt
from train import train, test_model
import numpy as np

def custom_accuracy(y_true, y_pred, tolerance=0.01):
    correct_predictions = torch.abs(y_true - y_pred) <= tolerance
    accuracy = torch.mean(correct_predictions.float())
    return accuracy

def main():
    now = datetime.datetime.now()
    logdir = '../trained_models/' + now.strftime('%Y-%m-%d-%H%M')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    config = config_reader.ConfigLoader('../configs/params.config')
    TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, TEST_BATCH_SIZE = config.get_batch_sizes()

    TEST_BATCH_SIZE = 100
    # ------------------------------------------------------------------------------------------------------------------
    # Load Training and Validation Data first
    # ------------------------------------------------------------------------------------------------------------------

    (training_data_file_name, training_labels_file_name, validation_data_file_name,
     validation_labels_file_name, test_data_file_name, test_labels_file_name) = config.get_file_names()

    q_vector, test_intensities, test_parameters, _, _ = data_preprocessing(
        data_file_name=test_data_file_name,
        labels_file_name=test_labels_file_name,
        q_max=117,
        noise_fct=experimental_bg)

    test_dataset = CustomDataset(torch.tensor(test_parameters, dtype=torch.float32),
                                 torch.tensor(test_intensities, dtype=torch.float32))

    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    number_of_parameters = test_parameters.shape[1]

    model_paths = ["../trained_models/forwardCNN_7param_6/SANS_CNN.pth",
                "../trained_models/forwardCNN_7param_7/SANS_CNN.pth",
                   "../trained_models/forwardCNN_7param_8/SANS_CNN.pth",
                   "../trained_models/forwardCNN_7param_9/SANS_CNN.pth"
                   ]

    models = [torch.load(model_path) for model_path in model_paths]
    forward_model = EnsembleModel(models)
    forward_model.eval()

    test_output = []
    test_target = []
    with torch.no_grad():
        for test_batch_idx, (test_inputs, test_targets) in enumerate(test_dataloader):
            test_outputs = forward_model(test_inputs)
            test_output.append(test_outputs)
            test_target.append(test_targets)

    test_output = torch.cat(test_output, dim=0)
    test_target = torch.cat(test_target, dim=0)


    # Compute the deviation for each data point
    deviation = torch.abs(test_output - test_target).sum(
        dim=1)  # Assuming 2D tensors; adjust if necessary

    # Find the index of the highest deviation
    index_of_max_deviation = torch.argmax(deviation).item()

    print(index_of_max_deviation)
    t_o = test_output.detach().numpy()
    t_t = test_target.detach().numpy()
    param = test_parameters[index_of_max_deviation, :]
    print(param)
    import matplotlib.pyplot as plt

    # plt.scatter(q_vector, test_intensities[index_of_max_deviation, :], s=1)
    plt.scatter(q_vector, t_o[index_of_max_deviation, :], s=1)
    plt.scatter(q_vector, t_t[index_of_max_deviation, :], s=1)
    plt.show()

    #
    mse_loss = torch.nn.MSELoss()

    ensemble_loss = mse_loss(test_output, test_target)
    ensemble_accuracy = custom_accuracy(test_output, test_target, tolerance=0.001)

    plot_forward_prediction_performance(test_output, test_target, q_vector, logdir,
                                        '/{}param_forward_prediction_performance'.format(number_of_parameters))

    # ------------------------------------------------------------------------------------------------------------------
    # Prints the performance parameters for the individual models and the ensemble one for comparison
    # comment out if you did not use a ensemble model.
    # ------------------------------------------------------------------------------------------------------------------

    single_model_loss = [mse_loss(*test_model(model, test_dataloader)) for model in models]
    single_model_acc = [custom_accuracy(*test_model(model, test_dataloader), tolerance=0.001) for model in models]
    #
    print('Ensemble Loss: {}, Ensemble Accuracy: {}'.format(ensemble_loss.item(), ensemble_accuracy))

    for loss, accuracy in zip(single_model_loss, single_model_acc):
        print('Single Model Loss: {}, Single Model Accuracy: {}'.format(loss.item(), accuracy))


if __name__ == '__main__':
    main()
