"""
Plot the convergence of the Modulation Transfer function (MTF) regression

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
from Derenzo_phantom.psf_mtf_library import fwhm_hermite_gaussian_1d, fwhm_plateau_polynomial_1d
from Derenzo_phantom.psf_mtf_regression import fit_mtf
from Derenzo_phantom.print_significant_digits import print_value_and_error, error_propagation
from axial_correlation import axial_correlation


def model_comparison():
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    derenzo_contrast_2d, derenzo_contrast_3d = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=True)

    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_true')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_energy')
    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BI-BI_energy')
    # axial_correlation(imgs[:, :, :, iterations == 200].squeeze())

    # derenzo_contrast_2d(np.mean(img_500, axis=-1))
    # derenzo_contrast_3d(img_500)

    # models = ['gaussian-expansion', 'lorentzian-raised', 'lorentzian-generalized', 'plateau-polynomial', 'plateau-sine', 'plateau-gaussian']
    models = ['gaussian', 'hermite-gaussian', 'lorentzian-generalized', 'plateau-polynomial']
    model_names = ['Gaussian', 'Hermite-Gaussian', 'Generalized Lorentzian', 'Plateau-polynomial']

    sum_squared_residuals = np.zeros((iterations.size, len(models)))
    mean_error = np.zeros(iterations.size)

    for ii in trange(iterations.size):
        wave_numbers, contrast_values, contrast_errors = derenzo_contrast_2d(np.mean(imgs[:, :, :, ii], axis=-1))
        # wave_numbers, contrast_values, contrast_errors = derenzo_contrast_3d(imgs[:, :, :, ii])
        mean_error[ii] = np.mean(contrast_errors)
        # mean_error[ii] = np.median(contrast_errors)
        # mean_error[ii] = np.percentile(contrast_errors, 50, method='linear')

        for jj in range(len(models)):
            fitted_function, _, _ = fit_mtf(wave_numbers, contrast_values, contrast_errors, model=models[jj], include_amplitude=False)

            # sum_squared_residuals[ii, jj] = np.sum((fitted_function(wave_numbers) - contrast_values) ** 2)
            # sum_squared_residuals[ii, jj] = np.sum(((fitted_function(wave_numbers) - contrast_values) / contrast_errors) ** 2)
            # sum_squared_residuals[ii, jj] = np.sum(((fitted_function(wave_numbers) - contrast_values) / mean_error[ii]) ** 2)
            sum_squared_residuals[ii, jj] = np.sum((fitted_function(wave_numbers) - contrast_values) ** 2) / mean_error[ii] ** 2

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8.7, 5))
    lines = ax.plot(iterations, sum_squared_residuals)
    [lines[ii].set_label(model_names[ii]) for ii in range(len(lines))]
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim()
    ax.set_ylim(4e1, 2.5e3)
    # ax.set_ylim(4e1, 2.5e4)
    ax.set_xlabel('Iteration number')
    ax.set_ylabel(r'$\sum\,_n\left[\mathrm{MTF}_\mathrm{fit}(k_n)-c_n\right]^2/\bar{\sigma}_c^2$')
    ax.legend(ncol=2, frameon=True)
    ax_twin = ax.twinx()
    ax_twin.plot(iterations, mean_error, color='k')
    ax_twin.set_ylim([0, 6e-2])
    ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_twin.set_ylabel(r'$\bar{\sigma}_c$')
    plt.show()

    return 0


def contrast_convergence(include_amplitude=False):
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    derenzo_contrast_2d, derenzo_contrast_3d = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=True)

    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_true')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_energy')
    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BI-BI_true')
    # axial_correlation(imgs[:, :, :, iterations == 200].squeeze())

    # derenzo_contrast_2d(np.mean(img_500, axis=-1))
    # derenzo_contrast_3d(img_500)

    frequency_samples = np.linspace(0, 4, 1000)

    if include_amplitude:
        p_opts = np.zeros((iterations.size, 3))
        p_errs = np.zeros((iterations.size, 3))
    else:
        p_opts = np.zeros((iterations.size, 2))
        p_errs = np.zeros((iterations.size, 2))

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=iterations[0], vmax=iterations[-1])

    # model = 'hermite-gaussian'
    model = 'plateau-polynomial'

    for ii in range(iterations.size):
        wave_numbers, contrast_values, contrast_errors = derenzo_contrast_2d(np.mean(imgs[:, :, :, ii], axis=-1))
        # wave_numbers, contrast_values, contrast_errors = derenzo_contrast_3d(imgs[:, :, :, ii])
        ax.errorbar(wave_numbers, contrast_values, yerr=contrast_errors, fmt='.', capsize=3, color=cmap(norm(iterations[ii])), alpha=0.25)

        fitted_function, p_opts[ii, :], p_errs[ii, :] = fit_mtf(wave_numbers, contrast_values, contrast_errors, model=model, include_amplitude=include_amplitude)
        ax.plot(frequency_samples, fitted_function(frequency_samples), color=cmap(norm(iterations[ii])))
        print(iterations[ii])
        print(p_opts[ii, :])
        # print_value_and_error(fwhm_hermite_gaussian_1d(*p_opts[ii, :]), error_propagation(fwhm_hermite_gaussian_1d, p_opts[ii, :], p_errs[ii, :]))
        # print_value_and_error(fwhm_plateau_polynomial_1d(*p_opts[ii, :]), error_propagation(fwhm_plateau_polynomial_1d, p_opts[ii, :], p_errs[ii, :]))

    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.add_lines(iterations, colors=cmap(norm(iterations)), linewidths=2)
    cbar.set_label('Iteration number')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 2.5)
    ax.set_yticks([0, 0.5, 1.])
    ax.set_xlabel(r'Wave number $k$ [1/mm]')
    ax.set_ylabel('MTF')
    plt.show()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(iterations, p_opts[:, 1], yerr=p_errs[:, 1], capsize=3, fmt='.', color='tab:orange')
    ax.plot(iterations, p_opts[:, 1], color='tab:orange')
    # ax.errorbar(iterations, p_opts[:, 2], yerr=p_errs[:, 2], capsize=3, fmt='.', linewidth=3, color='tab:orange')
    # ax.plot(iterations, p_opts[:, 2], linewidth=3, color='tab:orange')
    # ax.set_ylim(-0.1, 1.6)  # gaussian-expansion
    # ax.set_yticks([0, 0.5, 1., 1.5])  # gaussian-expansion
    ax.set_ylim(0.55, 1.25)
    ax.tick_params(axis='y', colors='tab:orange')
    ax.spines['left'].set_color('tab:orange')
    # ax.set_ylabel(r'$\alpha$ ( ) and amplitude ( )', color='tab:orange')  # gaussian-expansion
    ax.set_ylabel(r'$\alpha$', color='tab:orange')
    ax.set_xlabel('Iteration number')

    ax_twin = ax.twinx()
    ax_twin.errorbar(iterations, p_opts[:, 0], yerr=p_errs[:, 0], capsize=3, fmt='.', color='tab:blue')
    ax_twin.plot(iterations, p_opts[:, 0], color='tab:blue')
    ax_twin.set_xscale('log')
    # ax_twin.set_ylim(1.4, 3.1)  # gaussian-expansion
    # ax_twin.set_yticks([1.5, 2, 2.5, 3])  # gaussian-expansion
    ax_twin.set_ylim(0.2, 1.2)  # plateau-polynomial
    ax_twin.tick_params(axis='y', colors='tab:blue')
    ax_twin.spines['right'].set_color('tab:blue')
    # ax_twin.set_ylabel(r'$\sigma$ [mm]', color='tab:blue')  # gaussian-expansion
    ax_twin.set_ylabel(r'$k_{1/2}$ [mm$^{-1}$]', color='tab:blue')
    ax_twin.spines['left'].set_color('tab:orange')
    plt.show()

    return 0


if __name__ == "__main__":
    # model_comparison()
    contrast_convergence(include_amplitude=False)
