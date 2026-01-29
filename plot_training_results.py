import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import os


def kiel_diagram(y_test: np.ndarray,
                 predictions: np.ndarray,
                 label_names: list,
                 save_dir: str,
                 fe_h=False):
    """
    make a kiel diagram for test and predicted
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    binsx = np.linspace(4000, 7500, 100)
    binsy = np.linspace(0, 5, 100)

    if fe_h:
        H_weights, xedges, yedges = np.histogram2d(y_test[:, label_names.index('teff')],
                                                y_test[:, label_names.index('logg')],
                                                bins=[binsx, binsy],
                                                weights=y_test[:, label_names.index('m_h')])

        H_counts, _, _ = np.histogram2d(y_test[:, label_names.index('teff')],
                                        y_test[:, label_names.index('logg')],
                                        bins=(xedges, yedges))

        weighted_average = H_weights / H_counts


        res = ax1.imshow(weighted_average.T, origin='lower', aspect='auto',
                         extent=(binsx.min(), binsx.max(), binsy.min(), binsy.max()), cmap='inferno',
                        vmin=-1, vmax=0.3)
        plt.colorbar(res, label='[Fe/H]', ax=ax1)
    else:
        res = ax1.hist2d(y_test[:, label_names.index('teff')], y_test[:, label_names.index('logg')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
        plt.colorbar(res[-1], label='N', ax=ax1)
    ax1.grid()
    ax1.set_title('Test Data')
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('log(g)')
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    if fe_h:
        H_weights, xedges, yedges = np.histogram2d(predictions[:, label_names.index('teff')],
                                                predictions[:, label_names.index('logg')],
                                                bins=[binsx, binsy],
                                                weights=predictions[:, label_names.index('m_h')])

        H_counts, _, _ = np.histogram2d(predictions[:, label_names.index('teff')],
                                        predictions[:, label_names.index('logg')], bins=(xedges, yedges))

        weighted_average = H_weights / H_counts


        res = ax2.imshow(weighted_average.T, origin='lower', aspect='auto',
                         extent=(binsx.min(), binsx.max(), binsy.min(), binsy.max()), cmap='inferno',
                        vmin=-1, vmax=0.3)
        plt.colorbar(res, label='[Fe/H]', ax=ax2)
    else:
        res = ax2.hist2d(predictions[:, label_names.index('teff')], predictions[:, label_names.index('logg')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
        plt.colorbar(res[-1], label='N', ax=ax2)
    ax2.grid()
    ax2.set_title('Predictions')
    ax2.set_xlabel('Teff')
    ax2.set_ylabel('log(g)')
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    if fe_h:
        plt.savefig(f'{save_dir}/kiel_diagram_fe_h.png')
    else:
        plt.savefig(f'{save_dir}/kiel_diagram.png')
    plt.close()


def alpha_fe_plot(y_test: np.ndarray,
                  predictions: np.ndarray,
                  label_names: list,
                  save_dir: str):
    """
    Make plot of alpha/M vs Fe/H for test and predicted
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    binsx = np.linspace(-2, 0.5, 100)
    binsy = np.linspace(-0.4, 0.4, 100)

    res = ax1.hist2d(y_test[:, label_names.index('m_h')],
                     y_test[:, label_names.index('alpha_h')] - y_test[:, label_names.index('m_h')],
                     bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    plt.colorbar(res[-1], label='N', ax=ax1)
    ax1.grid()
    ax1.set_title('Test Data')
    ax1.set_xlabel('Fe/H')
    ax1.set_ylabel('alpha/M')

    res = ax2.hist2d(predictions[:, label_names.index('m_h')],
                     predictions[:, label_names.index('alpha_h')] - predictions[:, label_names.index('m_h')],
                     bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    plt.colorbar(res[-1], label='N', ax=ax2)
    ax2.grid()
    ax2.set_title('Predictions')
    ax2.set_xlabel('Fe/H')
    ax2.set_ylabel('alpha/M')
    plt.savefig(f'{save_dir}/alpha_m_vs_fe_h.png')
    plt.close()


if __name__ == '__main__':
    # load the params from saved model
    save_dir = 'nmf_joint_results_with_scatter_K32'
    res = np.load(f'{save_dir}/joint_model_results.npz')
    inferred_labels = res['inferred_labels']
    true_labels = res['true_labels']
    
    output_dir = 'nmf_joint_results_with_scatter_K32'
    label_names = ['teff', 'logg', 'm_h', 'alpha_h']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    # kiel diagram
    kiel_diagram(true_labels,
                 inferred_labels,
                 label_names,
                 output_dir,
                 fe_h=False)

    # keil diagram with Fe/H
    kiel_diagram(true_labels,
                 inferred_labels,
                 label_names,
                 output_dir,
                 fe_h=True)

    # alpha/M vs Fe/H
    alpha_fe_plot(true_labels,
                  inferred_labels,
                  label_names,
                  output_dir)
