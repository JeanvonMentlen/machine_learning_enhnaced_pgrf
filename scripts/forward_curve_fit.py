import datetime
import os
import timeit

import torch
import optuna
import numpy as np
from data_pipeline import load_data, log_normalization, ref_rescaling, ref_unscaling, experimental_bg
from performance_plotting import plot_fit_prediction_excerpt, plot_fitted_labels
import pandas as pd
from tqdm import tqdm
from models import EnsembleModel

from plurigaussianrandomfield import PluriGaussianRandomField

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def objective(trial, model, target_data):
    #-------------------------------------------------------------------------------------------------------------------
    # select the parameters you'd like to fit according to the model you are using. Since the 7 and 8 param models
    # predict the PGRF curve well, you can use them and set certain values constant to reduce the number of fit parameters.
    #-------------------------------------------------------------------------------------------------------------------

    int_const = trial.suggest_float('int_const', 0, 1)
    phi_A = trial.suggest_float('phi_A', 0, 1)  # 0.89
    vol_factor = trial.suggest_float('vol_factor', 0.31, .31)  # 0.31
    beta_angle = trial.suggest_float('beta_angle', 0.91, 0.91)  # 0.91
    l_z = trial.suggest_float('l_z', 0, 1)
    l_y = trial.suggest_float('l_y', 0, 1)
    porod_const = trial.suggest_float('porod_const', 0, 0)  # 0
    d_z = trial.suggest_float('d_z', 0, 1)  # 0.8
    rho_A = trial.suggest_float('rho_A', 0.11999, 0.11999)
    rho_B = trial.suggest_float('rho_B', 0.57999, 0.57999)
    rho_S = trial.suggest_float('rho_S', 0.02221, 0.02221)

    # labels = [int_const, phi_A, vol_factor, beta_angle, l_z, l_y, d_z]
    #labels = [int_const, phi_A, vol_factor, beta_angle, l_z, l_y, porod_const, d_z]
    labels = [int_const, phi_A, vol_factor, beta_angle, l_z, l_y, porod_const, d_z, rho_A, rho_B, rho_S]
    # labels = [int_const, l_z, l_y, d_z]
    # labels = [int_const, phi_A, l_z, l_y, porod_const]

    input_tensor = torch.tensor(labels).unsqueeze(0).unsqueeze(2)

    output = model(input_tensor)

    # Create a tensor of ones with length 117; changing the weights and setting the range depends on the importance
    # of certain features.
    weights = torch.ones(117)
    weights[50:97] = 6

    # alternatively use simple MSE loss
    # mse_loss = torch.nn.MSELoss()
    # loss = mse_loss(target_data, output)

    loss = weighted_mse_loss(output, target_data, weights)

    return loss.item()

def plot_result(best_parameter, target, q_vector, model, logdir, curve_nr, reference=None):
    params = [best_parameter[key] for key in best_parameter.keys()]
    input_tensor = torch.tensor(params).unsqueeze(0).unsqueeze(2)

    prediction = model(input_tensor)

    target_np = target.detach().numpy()
    prediction = prediction.detach().numpy()

    plot_fit_prediction_excerpt(val_output=prediction, val_target=target_np, q_vector=q_vector,
                                logdir=logdir, filename='/curve_{}'.format(curve_nr), reference=reference)

def restore_labels(label_dict):
    labels = np.array([label_dict[key] for key in label_dict.keys()])
    if len(labels) == 4:
        #int_const, phi_A, l_z, l_y
        label_mean = np.array([0.79839, 0.19018, 1.69884, 6.49894])
        label_std = np.array([0.40337, 0.08098, 0.74973, 2.03006])
        label_max = np.array([1.7927, 1.72643, 1.73552, 1.72462])
        label_min = np.array([-1.73138, -1.73095, -1.73242, -1.72358])

    if len(labels) == 5:
        #int_const, phi_A, l_z, l_y, porod_const
        label_mean = np.array([0.79839, 0.19018, 1.69884, 6.49894,0.00551])
        label_std = np.array([0.40337, 0.08098, 0.74973, 2.03006,0.00260])
        label_max = np.array([1.7927, 1.72643, 1.73552, 1.72462,1.72630])
        label_min = np.array([-1.73138, -1.73095, -1.73242, -1.72358,-1.73545])

    elif len(labels) == 7:
        # "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z"
        label_mean = np.array([0.80017,0.18973,1.24991,60.01523,1.69987,6.49782,6.99945])
        label_std = np.array([0.40399,0.08091,0.43268,17.34423,0.75029,2.02109,1.73251])
        label_max = np.array([1.73231,1.73369,1.73355,1.72880,1.73283,1.73282,1.73189 ])
        label_min = np.array([-1.73314,-1.72714,-1.73315,-1.73056,-1.73249,-1.73066,-1.73125])


    elif len(labels) == 8:
        # "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "porod const", "d_z",
        label_mean = np.array([0.79954,0.19020,1.24856,59.96877,1.69849,6.49420,0.05003,7.00484])
        label_std = np.array([0.40470,0.08073,0.43338,17.34713,0.74999,2.02008,0.02882,1.73518])
        label_max = np.array([1.73079,1.73168,1.73390,1.73120,1.73539,1.73548,1.73366,1.72612])
        label_min = np.array([-1.72849,-1.73669,-1.72727,-1.72755,-1.73135,-1.72974,-1.73567,-1.73171])

    elif len(labels) == 11:
        label_mean = np.array([0.79871, 0.18973, 1.25057, 60.07016, 1.70066, 6.74618, 0.00550, 6.99025, 19470320056.15366, 55001633944.08221, 19548339674.88811])
        label_std = np.array([0.40430, 0.08066, 0.43321, 17.30912, 0.75122, 1.87690, 0.00260, 1.73459, 12982155669.17257, 8657617652.08209, 12990495374.66872])
        label_max = np.array([1.73459, 1.73901, 1.72993, 1.72913, 1.72962, 1.73360, 1.72863, 1.73511, 1.73543, 1.73237, 1.72828])
        label_min = np.array([-1.72815, -1.73219, -1.73255, -1.73724, -1.73139, -1.72941, -1.73201, -1.72388, -1.73084, -1.73268, -1.73574])

    unscaled_label = ref_unscaling(labels, label_max, label_min, 0, 1)

    restored_label_list = unscaled_label * label_std + label_mean

    return dict(zip(label_dict.keys(), restored_label_list))


def get_pgrf_intensity(pgrf, opt_params, scaling_ref):
    intensity = pgrf.calculate_PGRF_intensity(int_const=opt_params['int_const'], phi_A=opt_params['phi_A'],
                                              vol_factor=opt_params['vol_factor'], beta_angle=opt_params['beta_angle'],
                                              l_z=opt_params['l_z'], l_y=opt_params['l_y'], b=opt_params['b'],
                                              porod_const=opt_params['porod_const'], d_y=opt_params['d_y'],
                                              d_z=opt_params['d_z'])
    data_file = np.loadtxt('../measurement_data/Cell2Curve5.txt', delimiter='\t', skiprows=1)
    exp_bg = data_file[:117, 1].transpose()

    return ref_rescaling(log_normalization(intensity + exp_bg), scaling_ref, 0, 1)



def main():
    pgrf = None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    now = datetime.datetime.now()
    logdir = '../fit/' + now.strftime('%Y-%m-%d-%H%M')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # model_paths = ["../trained_models/forwardCNN_7param_6/SANS_CNN.pth",
    #             "../trained_models/forwardCNN_7param_7/SANS_CNN.pth",
    #                "../trained_models/forwardCNN_7param_8/SANS_CNN.pth",
    #                "../trained_models/forwardCNN_7param_9/SANS_CNN.pth"
    #                ]

    #model_paths = ["../trained_models/forwardCNN_8param_paper_1/SANS_CNN.pth",
    #               "../trained_models/forwardCNN_8param_paper_2/SANS_CNN.pth",
    #           "../trained_models/forwardCNN_8param_paper_3/SANS_CNN.pth",
    #           "../trained_models/forwardCNN_8param_paper_4/SANS_CNN.pth"
    #           ]

    model_paths = ["../trained_models/forwardCNN_11param_including_SLD/SANS_CNN.pth"
                   ]

    models = [torch.load(model_path) for model_path in model_paths]
    model = EnsembleModel(models)
    model.eval()

    #-------------------------------------------------------------------------------------------------------------------
    # the reference values correspond to the min and max value of the dataset used to train the model after taking the
    # log. It is necessary to normalize the real data in the same way as the training dataset.
    #-------------------------------------------------------------------------------------------------------------------

    # 4 param
    # ref = [-1.1152849560998728, 8.388058380653916]
    # 7 param
    # ref = [-1.1188989782619472, 9.10524104764222]
    # 8 param
    #ref = [-1.120146732836264, 9.269371525308037]
    # 11 param
    ref = [-1.1205170400166378, 9.1170336372312]



    q_vector, target_curves = load_data('../measurement_data/Cell2Full.txt')

    #-------------------------------------------------------------------------------------------------------------------
    # PGRF for reference; comment out if you do not want to plot the pgrf curve to check if your model predicts the
    # curve well
    #-------------------------------------------------------------------------------------------------------------------
    pgrf = PluriGaussianRandomField(q_vector, [0, 120])

    target_curves = ref_rescaling(log_normalization(target_curves), ref, 0, 1)

    #-------------------------------------------------------------------------------------------------------------------
    # Full label list: "curve_nr", "fit_error", "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z",
    # 'porod_const',
    #-------------------------------------------------------------------------------------------------------------------
    # 7 param
    # df = pd.DataFrame(columns=["curve_nr", "fit_error", "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "d_z"])
    # 8 param
    #df = pd.DataFrame(columns=["curve_nr", "fit_error", "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "porod_const", "d_z"])
    # 11 param
    df = pd.DataFrame(
        columns=["curve_nr", "fit_error", "int_const", "phi_A", "vol_factor", "beta_angle", "l_z", "l_y", "porod_const",
                 "d_z", "rho_A", "rho_B", "rho_S"])

    for curve_nr in tqdm(range(target_curves.shape[0])):
    #for curve_nr in tqdm(range(10)):
        target = torch.tensor(target_curves[curve_nr, :117].transpose()).unsqueeze(0)

        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, model, target), n_trials=10000)

        print(f"Curve Number: {curve_nr}, Best Trial: {study.best_trial.number}")

        restored_labels = restore_labels(study.best_params)
        restored_labels['curve_nr'] = curve_nr
        restored_labels['fit_error'] = study.best_value
        df = pd.concat([df, pd.DataFrame([restored_labels])], ignore_index=True)

        if pgrf is not None:
            # depending on the model used, you need to specify certain values it was not trained on for the PGRF
            restored_labels['d_y'] = 150
            restored_labels['b'] = 5.8
            # restored_labels['porod_const'] = 0
            # restored_labels['phi_A'] = 0.3
            # restored_labels['vol_factor'] = 1
            # restored_labels['beta_angle'] = 85
            # restored_labels['d_z'] = 8


            pgrf_int = get_pgrf_intensity(pgrf, restored_labels, ref)

            plot_result(
                best_parameter=study.best_params,
                target=target,
                q_vector=q_vector[:117],
                model=model,
                logdir=logdir,
                curve_nr=curve_nr,
                reference=pgrf_int[:117]
            )

        else:
            plot_result(
                best_parameter=study.best_params,
                target=target,
                q_vector=q_vector[:117],
                model=model,
                logdir=logdir,
                curve_nr=curve_nr
                )

    plot_fitted_labels(df, logdir)
    df.to_csv(logdir+'/fited_labels.csv', sep='\t', index=False)

if __name__ == "__main__":
    main()
