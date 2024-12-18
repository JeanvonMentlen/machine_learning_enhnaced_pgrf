import datetime
import os
import torch
from train import mae, mse
import optuna
import numpy as np
from models import EnsembleModel
from data_pipeline import load_data, log_normalization, ref_rescaling, ref_unscaling, experimental_bg
from performance_plotting import plot_fit_prediction_excerpt, plot_fitted_labels
import pandas as pd
from tqdm import tqdm
from plurigaussianrandomfield import PluriGaussianRandomField
import sys

def restore_labels(labels, label_names):
    if len(label_names) == 4:
        #int_const, phi_A, l_z, l_y
        label_mean = np.array([0.79839, 0.19018, 1.69884, 6.49894])
        label_std = np.array([0.40337, 0.08098, 0.74973, 2.03006])
        label_max = np.array([1.7927, 1.72643, 1.73552, 1.72462])
        label_min = np.array([-1.73138, -1.73095, -1.73242, -1.72358])

    if len(label_names) == 5:
        #int_const, phi_A, l_z, l_y, porod_const
        label_mean = np.array([0.79839, 0.19018, 1.69884, 6.49894,0.00551])
        label_std = np.array([0.40337, 0.08098, 0.74973, 2.03006,0.00260])
        label_max = np.array([1.7927, 1.72643, 1.73552, 1.72462,1.72630])
        label_min = np.array([-1.73138, -1.73095, -1.73242, -1.72358,-1.73545])

    elif len(label_names) == 7:
        # "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z"
        label_mean = np.array([0.79857,0.18973,1.25038,60.04603,1.20079,6.49920,7.00026 ])
        label_std = np.array([0.40369,0.08087,0.43221,17.36238,0.46186,2.02206,1.73576])
        label_max = np.array([1.73755,1.73443,1.73434,1.72522,1.73042,1.73130,1.72819])
        label_min = np.array([-1.73043,-1.72775,-1.73610,-1.73053,-1.73384,-1.73051,-1.72848])


    elif len(label_names) == 8:
        # "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z", "porod const"
        label_mean = np.array([0.79929,0.18988,1.24854,59.95868,1.69798,6.50024,1.05152,0.00551 ])
        label_std = np.array([0.40352,0.08072,0.43276,17.33034,0.75014,2.02131,0.54687,0.00260 ])
        label_max = np.array([1.73647,1.73581,1.73642,1.73345,1.73570,1.73144,1.73439,1.72630 ])
        label_min = np.array([-1.73296,-1.73286,-1.72969,-1.72865,-1.73032,-1.73167,-1.73985,-1.73545])

    unscaled_label = ref_unscaling(labels, label_max, label_min, 0, 1)

    restored_label_list = unscaled_label * label_std + label_mean

    return dict(zip(label_names, restored_label_list.transpose()))


def get_pgrf_intensity(pgrf, opt_params, scaling_ref):
    predicted_intensities = []
    for ii in tqdm(range(len(opt_params['int_const']))):
        intensity = pgrf.calculate_PGRF_intensity(int_const=opt_params['int_const'][ii], phi_A=opt_params['phi_A'][ii],
                                                  vol_factor=opt_params['vol_factor'][ii], beta_angle=opt_params['beta_angle'][ii],
                                                  l_z=opt_params['l_z'][ii], l_y=opt_params['l_y'][ii], b=opt_params['b'][ii],
                                                  porod_const=opt_params['porod_const'][ii], d_y=opt_params['d_y'][ii],
                                                  d_z=opt_params['d_z'][ii])
        data_file = np.loadtxt('../measurement_data/Cell2Curve5.txt', delimiter='\t', skiprows=1)
        exp_bg = data_file[:117, 1].transpose()
        predicted_intensities.append(ref_rescaling(log_normalization(intensity + exp_bg), scaling_ref, 0, 1))

    return np.array(predicted_intensities)

def mean_squared_error(y_true, y_pred):
    # Calculate the squared differences
    squared_diff = np.square(y_true - y_pred)

    # Calculate the mean along axis=0
    mse = np.mean(squared_diff, axis=1)

    return mse

def main():
    pgrf = None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    now = datetime.datetime.now()
    logdir = 'fit/' + now.strftime('%Y-%m-%d-%H%M')
    logdir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model_path = "../trained_models/inverseCNN_8param/SANS_CNN.pth"
    model = torch.load(model_path)
    model.eval()

    model_paths = ["../trained_models/forwardCNN_8param_paper_1/SANS_CNN.pth",
                   "../trained_models/forwardCNN_8param_paper_2/SANS_CNN.pth",
                   "../trained_models/forwardCNN_8param_paper_3/SANS_CNN.pth",
                   "../trained_models/forwardCNN_8param_paper_4/SANS_CNN.pth"
                   ]

    models = [torch.load(model_path) for model_path in model_paths]
    fwd_model = EnsembleModel(models)
    fwd_model.eval()


    # 4 param
    # ref = [-1.0990196445812446, 7.870525946405369]
    # 7 param
    #ref = [-1.11672903964104, 9.104947523853946]
    # 8 param
    ref = [-1.124917754610575, 9.22349680266662]


    q_vector, target_curves = load_data('../measurement_data/Cell2Full.txt')

    # PGRF for reference
    pgrf = PluriGaussianRandomField(q_vector, [0, 120])

    target_curves = ref_rescaling(log_normalization(target_curves), ref, 0, 1)
    target_curves = target_curves[:, :]
    np.savetxt(logdir+'/normalized_target_curves.txt', target_curves, delimiter='\t')

    # df = pd.DataFrame(columns=["curve_nr", "fit_error", "int_const", "l_z", "l_y", "d_z"])

    #labels = ["int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z"]
    #labels = ["int_const", 'phi_A', "l_z", "l_y"]
    labels = ["int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "porod_const", "d_z"]

    target_curves = torch.tensor(target_curves.tolist()).unsqueeze(2)

    predicted_labels = model(target_curves)
    predicted_intensities = fwd_model(predicted_labels.unsqueeze(2))

    # ------------------------------------------------------------------------------------------------------------------
    # Select the labels that were not predicted by the inverseCNN
    # ------------------------------------------------------------------------------------------------------------------

    predicted_labels_restored = restore_labels(predicted_labels.detach().numpy(), labels)
    predicted_labels_restored['curve_nr'] = np.linspace(0, target_curves.shape[0]-1, target_curves.shape[0])
    #predicted_labels_restored['d_y'] = np.ones_like(predicted_labels_restored['curve_nr']) * 150
    #predicted_labels_restored['d_z'] = np.ones_like(predicted_labels_restored['curve_nr']) * 8
    #predicted_labels_restored['b'] = np.ones_like(predicted_labels_restored['curve_nr']) * 5.8
    #predicted_labels_restored['porod_const'] = np.ones_like(predicted_labels_restored['curve_nr']) * 0.005
    #predicted_labels_restored['phi_A'] = np.ones_like(predicted_labels_restored['curve_nr']) * 0.3
    #predicted_labels_restored['vol_factor'] = np.ones_like(predicted_labels_restored['curve_nr']) * 1
    #predicted_labels_restored['beta_angle'] = np.ones_like(predicted_labels_restored['curve_nr']) * 85
    predicted_labels_restored['error'] = mean_squared_error(target_curves.detach().numpy()[:, :, 0], predicted_intensities.detach().numpy())


    # print(target_curves.detach().numpy()[20, :, 0] -  predicted_intensities.detach().numpy()[20, :])
    # pgrf_int = get_pgrf_intensity(pgrf, predicted_labels_restored, ref)
    # np.savetxt(logdir+'/predicted_curves.txt', pgrf_int, delimiter='\t')
    np.savetxt(logdir+'/predicted_curves.txt', predicted_intensities.detach().numpy(), delimiter='\t')

    ordered_col_names = ['curve_nr', 'int_const', "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z", "porod_const"]
    df = pd.DataFrame.from_dict(predicted_labels_restored)
    df = df[ordered_col_names]
    df.to_csv(logdir+'/predicted_parameters.txt', sep='\t')


if __name__ == "__main__":
    main()
