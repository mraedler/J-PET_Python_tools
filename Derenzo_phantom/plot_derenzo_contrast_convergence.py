"""
Plot the convergence of the derenzo contrast analysis

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# Auxiliary functions
from CASToR.read_interfile import read_interfile
from CASToR.vis import vis_3d
from get_derenzo_image import get_ground_truth_derenzo_image, load_derenzo_image
from get_derenzo_contrast import get_derenzo_contrast_function
from psf_gaussian_expansion import fit_psf_frequency_domain_1d
from axial_correlation import axial_correlation


def main():
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    derenzo_contrast_2d, derenzo_contrast_3d = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=True)

    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_true')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_energy')
    # axial_correlation(imgs[:, :, :, iterations == 200].squeeze())

    # derenzo_contrast_2d(np.mean(img_500, axis=-1))
    # derenzo_contrast_3d(img_500)

    frequency_samples = np.linspace(0, 4, 1000)

    p_opts = np.zeros((iterations.size, 3))
    p_errs = np.zeros((iterations.size, 3))

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=iterations[0], vmax=iterations[-1])

    for ii in trange(iterations.size):
        wave_numbers, contrast_values, contrast_errors = derenzo_contrast_2d(np.mean(imgs[:, :, :, ii], axis=-1))
        # wave_numbers, contrast_values, contrast_errors = derenzo_contrast_3d(imgs[:, :, :, ii])
        ax.errorbar(wave_numbers, contrast_values, yerr=contrast_errors, fmt='.', capsize=3, color=cmap(norm(iterations[ii])), alpha=0.25)

        fitted_function, p_opts[ii, :], p_errs[ii, :] = fit_psf_frequency_domain_1d(wave_numbers, contrast_values, contrast_errors, include_scaling=True)
        ax.plot(frequency_samples, fitted_function(frequency_samples), color=cmap(norm(iterations[ii])))

    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.add_lines(iterations, colors=cmap(norm(iterations)), linewidths=2)
    cbar.set_label('Iteration number')

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 2.5)

    ax.set_yticks([0, 0.5, 1.])

    ax.set_xlabel(r'Wave number $k$ [1/mm]')
    ax.set_ylabel('MTF')

    plt.show()

    # print(np.hstack((iterations[:, np.newaxis], p_opts)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(iterations, p_opts[:, 1], yerr=p_errs[:, 1], capsize=3, fmt='.', color='tab:orange')
    ax.plot(iterations, p_opts[:, 1], color='tab:orange')
    ax.errorbar(iterations, p_opts[:, 2], yerr=p_errs[:, 2], capsize=3, fmt='.', linewidth=3, color='tab:orange')
    ax.plot(iterations, p_opts[:, 2], linewidth=3, color='tab:orange')
    ax.set_ylim(-0.1, 1.6)
    ax.set_yticks([0, 0.5, 1., 1.5])
    ax.tick_params(axis='y', colors='tab:orange')
    ax.spines['left'].set_color('tab:orange')
    ax.set_ylabel(r'$\alpha$ ( ) and scale ( )', color='tab:orange')
    ax.set_xlabel('Iteration number')

    ax_twin = ax.twinx()
    ax_twin.errorbar(iterations, p_opts[:, 0], yerr=p_errs[:, 0], capsize=3, fmt='.', color='tab:blue')
    ax_twin.plot(iterations, p_opts[:, 0], color='tab:blue')
    ax_twin.set_xscale('log')
    ax_twin.set_ylim(1.4, 3.1)
    ax_twin.set_yticks([1.5, 2, 2.5, 3])
    ax_twin.tick_params(axis='y', colors='tab:blue')
    ax_twin.spines['right'].set_color('tab:blue')
    ax_twin.set_ylabel(r'$\sigma$ [mm]', color='tab:blue')

    ax_twin.spines['left'].set_color('tab:orange')

    # ax.grid()

    plt.show()

    return 0


if __name__ == "__main__":
    main()
