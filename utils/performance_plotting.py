import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from tqdm import tqdm


def plot_training_performance(training_history, validation_history, num_epochs, logdir, filename):
    import matplotlib.pyplot as plt
    epochs = range(num_epochs)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    ax[0].plot(epochs, training_history[:, 0], label='Training Loss', linewidth=3, c='#1F407A')
    ax[0].plot(epochs, validation_history[:, 0], label='Validation Loss', linewidth=3, c='#A8322D')
    ax[0].set_xlabel('Epoch', size=36, labelpad=10)
    ax[0].set_ylabel('MSE', size=36, labelpad=10)
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='both', which='major', labelsize=32)
    ax[0].legend(fontsize=20)

    ax[1].plot(epochs, training_history[:, 1], label='Training Acc', linewidth=3, c='#1F407A')
    ax[1].plot(epochs, validation_history[:, 1], label='Validation Acc', linewidth=3, c='#A8322D')
    ax[1].set_xlabel('Epoch', size=36, labelpad=10)
    ax[1].set_ylabel('Accuracy', size=36, labelpad=10)
    ax[1].tick_params(axis='both', which='major', labelsize=32)
    ax[1].legend(fontsize=20)

    plt.tight_layout()

    plt.savefig(logdir+filename+'.png')
    plt.savefig(logdir+filename+'.svg')

    np.savetxt(logdir+'/training_performance.npy', np.concatenate([training_history[:, 0], validation_history[:, 0], training_history[:, 1], validation_history[:, 1]]), delimiter='\t')

    # plt.show()

def plot_inverse_prediction_performance(val_output, val_target, logdir, filename, title=None):

    n = val_output.shape[1]

    fig, ax = plt.subplots(1, n)
    fig.set_size_inches(n*4, 4)
    plt.subplots_adjust(hspace=0.5)

    if n == 7:
        labels = ['int_const', 'phi_A', 'vol_factor', 'beta_angle', 'l_z', 'l_y', 'd_z']
    elif n == 6:
            labels = ['int_const', 'phi_A', 'l_z', 'l_y', 'd_z', 'porod_const']
    elif n == 8:
            labels = ['int_const', 'phi_A', 'vol_factor', 'beta_angle', 'l_z', 'l_y', 'd_z', 'porod_const']
    elif n == 5:
        labels = ['int_const', 'phi_A', 'l_z', 'l_y', 'porod_const']
    else:
        # labels = ['int_const', 'l_z', 'l_y', 'd_z']
        labels = ['int_const', 'phi_A','l_z', 'l_y']

    for ii in tqdm(range(n)):
       ax[ii % n].set_title(labels[ii], size=20)
       ax[ii % n].scatter(val_target[:, ii], val_output[:, ii], c='r', s=0.1, lw=1, ec='black')
       ax[ii % n].set_xlabel('Target', size=16)
       ax[ii % n].tick_params(axis='both', which='major', labelsize=16)
       ax[ii % n].plot([0, 1], [0, 1], lw=1, color='#0000FF')
       if ii % n != 0:
           ax[ii % n].set_yticks([])

    ax[0].set_ylabel('Prediction', size=16)
    plt.savefig(logdir+filename+'.png', dpi=300, bbox_inches='tight')
    plt.savefig(logdir+filename+'.svg', bbox_inches='tight')
    plt.show()


def plot_forward_prediction_performance(val_output, val_target, q_vector, logdir, filename, title=None):
    import matplotlib.pyplot as plt

    array_length = len(q_vector)
    indices = np.linspace(0, array_length - 1, 6, dtype=int)
    tick_labels = [np.round(q_vector[ii], 3) for ii in indices]
    tick_locations = np.linspace(0, 1, 6)

    x = val_target.flatten()
    y = val_output.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.scatter(x, y, c='k', s=1, alpha=0.8)

    ax.set_title('Q-Values', size=68)
    ax.plot(0, 1, lw=3, color='#0000FF')
    ax.set_xlabel('Target', size=62, labelpad=20)
    ax.set_ylabel('Prediction', size=62, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.set_xlim(1, 0)
    # Move the x-axis to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel('Target', size=62, labelpad=20)

    # Create a new axis with limits and labels from 1 to 0
    top_ax = ax.twiny()
    top_ax.set_xlim(1, 0)
    top_ax.tick_params(axis='x', which='major', labelsize=42)
    top_ax.xaxis.set_major_locator(FixedLocator(tick_locations))
    top_ax.set_xticklabels(tick_labels)

    if title is not None:
        plt.title(title)

    plt.savefig(logdir+filename+'.png', bbox_inches='tight')
    # plt.savefig(logdir+filename+'.svg', bbox_inches='tight')
    #fig.show()


def plot_forward_prediction_excerpt(val_output, val_target, q_vector, logdir, filename):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a, b = 5, 5
    n = a * b


    fig, ax = plt.subplots(a, b)
    fig.set_size_inches(20, 20)

    labels = ['Target', 'Prediction']
    colors = ['k', '#A8322D']
    handles = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]

    for ii in range(n):
        ax[ii // a, ii % b].scatter(q_vector, val_target[ii, :], c='k', s=1)
        ax[ii // a, ii % b].scatter(q_vector, val_output[ii, :], c='#A8322D', s=1)
        # ax[ii // a, ii % b].set_yscale('log')
        ax[ii // a, ii % b].set_xscale('log')

    plt.legend(handles, labels, fontsize=24, bbox_to_anchor=(-0.75, -0.75), loc='lower left')
    plt.savefig(logdir+filename+'.png')
    plt.savefig(logdir+filename+'.svg')
    #fig.show()

def plot_fit_prediction_excerpt(val_output, val_target, q_vector, logdir, filename, title=None, reference=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)

    ax.scatter(q_vector, val_target[0, :], c='k', s=3, label='target')
    ax.scatter(q_vector, val_output[0, :], c='#A8322D', s=3, label='fit')
    if reference is not None:
        ax.scatter(q_vector, reference, c='#007A96', s=3, label='PGRF+BG')
    # ax.set_yscale('log')
    ax.set_xscale('log')

    if title is not None:
        plt.title(title)

    plt.legend()

    plt.savefig(logdir + filename + '.png')
    plt.close()
    # plt.savefig(logdir + filename + '.svg')


def plot_fitted_labels(df, logdir):

    # Get all columns except 'curve_nr'
    columns = [col for col in df.columns if col != 'curve_nr']

    # Calculate the number of rows and columns for subplots
    n = len(columns)
    ncols = 2  # you can adjust this as needed
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))

    # Flatten the array of axes, to make iterating easier
    axs = axs.flatten()

    # Plot each column against 'curve_nr'
    for i, col in enumerate(columns):
        axs[i].plot(df['curve_nr'], df[col])
        axs[i].set_title(f'{col} vs curve_nr')

        if i == 0:
            axs[i].set_yscale('log')

    # Remove unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axs[j])

    # Automatically adjust subplot layouts so that there is no overlap between subplots
    plt.tight_layout()
    plt.savefig(logdir+'/fited_labels.png')
    plt.show()
