import numpy as np
from collections import OrderedDict
from numpy.random import uniform, poisson
import config_reader
from plurigaussianrandomfield import PluriGaussianRandomField
from tqdm import tqdm
import csv

config = config_reader.ConfigLoader('../configs/params.config')

test_data = np.genfromtxt('../measurement_data/data_generation_reference.txt')

noise_factor = 0

generation_data_file_name, generation_labels_file_name = config.get_generation_file_names()
number_of_training_samples = config.get_number_of_training_samples()

parameter_range = config.get_parameters()

def main():
    np.random.seed(config.get_random_seed())

    training_data, training_inputs, training_labels = create_data_set(number_of_training_samples)
    save_data_to_csv(training_data, generation_data_file_name)
    save_label_to_csv(training_inputs, training_labels, generation_labels_file_name)


def create_data_set(n_samples):
    intensities, input_data, input_label = calculate_scattering_curve(parameter_range, n_samples)

    training_data_output = np.zeros([len(test_data[:, 0]), n_samples + 1])
    training_data_output[:, 0] = test_data[:, 0]
    training_data_output[:, 1:] = intensities

    return training_data_output, input_data, input_label

def create_set_of_parameters(p):
    params = OrderedDict({
        'int_const': uniform(p['int_const_min'], p['int_const_max'], 1)[0],
        #'int_const': 1,

        'phi_A': uniform(p['phi_A_min'], p['phi_A_max'], 1)[0],
        #'phi_A': 0.3,

        'vol_factor': uniform(p['vol_factor_min'], p['vol_factor_max'], 1)[0],
        # 'vol_factor': 1,

        'beta_angle': uniform(p['beta_angle_min'], p['beta_angle_max'], 1)[0],
        # 'beta_angle': 85,

        'l_z': uniform(p['l_z_min'], p['l_z_max'], 1)[0],
        # 'l_z': 1,

        'l_y': uniform(p['l_y_min'], p['l_y_max'], 1)[0],
        # 'l_y': 9,

        # 'b': uniform(p['b_min'], p['b_max'], 1)[0],
        'b': 5.8,

        'porod_const': uniform(p['porod_const_min'], p['porod_const_max'], 1)[0],
        # 'porod_const': 0,

        'd_y': 150,
        # 'd_y': uniform(p['d_y_min'], p['d_y_max'], 1)[0],

        'd_z': uniform(p['d_z_min'], p['d_z_max'], 1)[0],
        # 'd_z': 8,

        'rho_A': np.round(uniform(p['rho_A_min'], p['rho_A_max'], 1)[0] * 1E+10, 0),
        # 'rho_A': 0.24E+10,
        'rho_B': np.round(uniform(p['rho_B_min'], p['rho_B_max'], 1)[0] * 1E+10, 0),
        # 'rho_B': 5.74E+10,
        'rho_S': np.round(uniform(p['rho_S_min'], p['rho_S_max'], 1)[0] * 1E+10, 0)
        # 'rho_S': -0.207E+10,

    })


    return params

def calculate_scattering_curve(parameter_range, n_samples):
    intensity_curves = np.zeros([len(test_data[:, 0]), n_samples])
    parameter_sets = np.zeros([n_samples, int(len(parameter_range.keys()) / 2)])
    q_vector = test_data[:, 0] * 10
    PGRF = PluriGaussianRandomField(q_vector, [np.min(test_data[:, 0]), np.max(test_data[:, 0])])
    params = {}

    for param_set in tqdm(range(n_samples)):
        params = create_set_of_parameters(parameter_range)
        intensity = PGRF.calculate_PGRF_intensity(int_const=params['int_const'], phi_A=params['phi_A'],
                                                  vol_factor=params['vol_factor'], beta_angle=params['beta_angle'],
                                                  l_z=params['l_z'], l_y=params['l_y'], b=params['b'],
                                                  porod_const=params['porod_const'], d_y=params['d_y'], d_z=params['d_z'],
                                                  rho_A=params['rho_A'], rho_B=params['rho_B'], rho_S=params['rho_S'])
        intensity_curves[:, param_set] = intensity
        parameter_sets[param_set, :] = np.array([params[key] for key in params], dtype=object).transpose()

    return intensity_curves, parameter_sets, params.keys()

def save_data_to_csv(data, file_name):
    number_of_samples = data.shape[1] - 1
    header = []

    for ii in range(number_of_samples):
        sample = ii + 1
        header += ['Scattering {}'.format(sample)]

    header = ['q'] + header

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(data)

def save_label_to_csv(label_data, labels, file_name):
    header = []

    for parameter in  labels:
        header += [parameter]

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(label_data)

if __name__ == '__main__':
    main()