"""

"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Auxiliary functions
from get_derenzo_image import get_ground_truth_derenzo_image
from get_derenzo_contrast import get_derenzo_contrast_function
from psf_gaussian_expansion import psf_frequency_domain_1d, fit_psf_frequency_domain_1d


def main():
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    img_peaks = np.sum(img_peaks, axis=-1)
    k_max = 2 * np.pi / (2 * (x_grid[1] - x_grid[0]))

    derenzo_contrast_2d, _ = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=False)
    wave_numbers, contrast_values, contrast_errors = derenzo_contrast_2d(img_peaks)
    fitted_function, p_opt, p_err = fit_psf_frequency_domain_1d(wave_numbers, contrast_values, contrast_errors, include_scaling=False)

    print(p_opt)
    print(p_err)

    wave_number_samples = np.linspace(0, 50, 1000)
    fit = fitted_function(wave_number_samples)

    # step_pos = k_max
    # step_width = 1
    # correction_factor = fit * smooth_step(step_pos - wave_number_samples, step_width) + smooth_step(wave_number_samples - step_pos, step_width)
    # fig, ax = plt.subplots()
    # ax.plot(wave_number_samples, fit)
    # ax.plot(wave_number_samples, correction_factor)
    # plt.show()
    # np.save(sys.path[0] + '/Pixelated_ground_truth/correction_wave_numbers.npy', wave_number_samples)
    # np.save(sys.path[0] + '/Pixelated_ground_truth/correction_factors.npy', correction_factor)

    x_lim = np.array([0, 20])
    y_lim = np.array([-0.2, 1.2])

    x_edges = np.linspace(0, 2.5, 50)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    x_grid, y_grid = np.meshgrid(x_edges, y_lim)

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots()
    ax.pcolormesh(x_grid, y_grid, psf_frequency_domain_1d(x_centers, 1.5, 1)[np.newaxis, :], vmin=0, vmax=3, cmap='binary')

    ax.errorbar(wave_numbers, contrast_values, yerr=contrast_errors, fmt='.', capsize=2, label='MTF data')
    ax.plot(wave_number_samples, fit, label=r'$\mathrm{MTF}(k_n)$ fit')
    ax.plot([k_max, k_max], y_lim, '--', color='k')

    ax.text(k_max, 1, r'$k_\mathrm{max}$', rotation='vertical', ha='right', va='center')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_yticks([0, 0.5, 1.])
    ax.set_xlabel(r'$k_n$ [1/mm]')
    ax.legend(frameon=False)
    plt.show()

    return 0


def smooth_step(x, alpha):
    # return np.arctan(x / alpha) / np.pi + 1 / 2
    return np.tanh(x / alpha) / 2 + 1 / 2


if __name__ == "__main__":
    main()
