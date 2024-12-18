import numpy as np
import scipy
from scipy.stats import tstd, zscore
import os
import sys
import matplotlib.pyplot as plt

def load_data(data_file_path, skiprows=0, q_max=None):
    data_file = np.loadtxt(data_file_path, delimiter='\t', skiprows=skiprows)
    q_vector = data_file[:, 0]
    intensity = data_file[:, 1:].transpose()

    if q_max is not None:
        q_vector = q_vector[:q_max]
        intensity = intensity[:, :q_max]

    return q_vector, intensity


def load_labels(labels_file_path):
    labels = np.loadtxt(labels_file_path, delimiter='\t')
    return labels


def normalize_labels(labels):
    # int_const	phi_A	vol_factor	beta_angle	l_z	l_y	b	porod_const	d_z	d_y
    means = scipy.mean(labels, axis=0)
    variance = tstd(labels, axis=0)
    normalized_labels = zscore(labels, axis=0)
    label_max = [np.max(normalized_labels[:, ii]) for ii in range(labels.shape[1])]
    label_min = [np.min(normalized_labels[:, ii]) for ii in range(normalized_labels.shape[1])]
    normalized_labels = normalized_labels[:, ~np.isnan(normalized_labels).any(axis=0)]

    for ii in range(normalized_labels.shape[1]):
        normalized_labels[:, ii] = rescaling(normalized_labels[:, ii], 0, 1)

    return normalized_labels, np.stack([means, variance, label_max, label_min])


def data_preprocessing(noise_fct, data_file_name, labels_file_name, q_max=167):
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', data_file_name))
    label_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', labels_file_name))
    q_vector, intensity = load_data(data_path, q_max=q_max)

    #if labels_file_name is not None:
    labels = load_labels(label_path)

    fig, ax = plt.subplots(1, 4)
    ax[0].hist(labels[:, 0])
    ax[1].hist(labels[:, 1])
    ax[2].hist(labels[:, 4])
    ax[3].hist(labels[:, 5])

    fig.show()

    normalized_labels, label_statistics = normalize_labels(labels)

    normalized_noisy_intensity = noise_fct(intensity, q_max)

    return q_vector, normalized_noisy_intensity.round(3), normalized_labels.round(3), label_statistics, labels

def data_postprocessing(labels, label_stats):
    if labels.shape[0] == 0:
        means = label_stats[0, :]
        variance = label_stats[1, :]
        restored_labels = np.zeros(labels.shape)

        for mean, var, label in zip (means, variance, range(labels.shape[1])):
            restored_labels[:, label] = var * labels[:, label] + mean

        return restored_labels

    else:
        restored_labels = np.ones((labels.shape[0], label_stats.shape[1]))
        ii = 0
        for col in range(label_stats.shape[1]):
            if np.isclose(label_stats[1, col], 0, atol=1e-8):
                restored_labels[:, col] *= label_stats[0, col]

            else:
                means = label_stats[0, col]
                variance = label_stats[1, col]
                restored_labels[:, col] *= variance * labels[:, ii]
                restored_labels[:, col] += means
                ii += 1
        return restored_labels


def apply_noise(data,):
    #add poisson noise
    #data = np.random.poisson(data)
    data += np.random.uniform(0, 0.2, data.shape[1])
    data = data.astype(float)
    return data

def zero_normalization(data):
    for index in range(data.shape[0]):
        data[index, :] += np.abs(data[index, :].min())
        data[index, :] /= data[index, :].max()
    return data


def remove_bad_data(data, label):
    # drop rows with sub 0 intensities
    clean_data = data[np.all(data > 0, axis=1), :]
    clean_labels = label[np.all(data > 0, axis=1), :label.shape[1]]
    return clean_data, clean_labels

def log_normalization(data):
    #set 0 to 1 before log normalization.
    min = np.nanmin(data)
    max = np.nanmax(data)
    data[np.where(data <= 0)] = np.nan
    data = np.log(data)
    #if not np.isfinite(data).all():
    #    raise ValueError

    return data

def rescaling(data, a, b):
    min = np.nanmin(data)
    max = np.nanmax(data)
    data = (b-a) * (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)) + a
    return data

def ref_rescaling(data, reference, a, b):
    min = np.nanmin(data)
    max = np.nanmax(data)
    data = (b - a) * (data - np.nanmin(reference)) / (np.nanmax(reference) - np.nanmin(reference)) + a
    return data

def ref_unscaling(scaled_data, max_val, min_val, a ,b):
    unscaled_data = (scaled_data - a) / (b - a) * (max_val - min_val) + min_val
    return unscaled_data

def poisson_noise(data):
    return rescaling(log_normalization(apply_noise(data)), 0, 1)

def noise_free(data, q_max=None):
    return rescaling(log_normalization(data), 0, 1)

def experimental_bg(data, q_max=None):
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', 'measurement_data/Cell2Curve5.txt'))

    data_file = np.loadtxt(data_path, delimiter='\t', skiprows=1)
    exp_bg = data_file[:, 1].transpose()
    if q_max is not None:
        exp_bg = exp_bg[:q_max]
    return rescaling(log_normalization(data + exp_bg), 0, 1)

def const_bg(data, q_max=None):
    const_bg = np.array([np.ones((data.shape[1])) * np.random.uniform(0.3, 0.35, 1) for ii in range(data.shape[0])])
    return rescaling(log_normalization(apply_noise(data) + const_bg), 0, 1)