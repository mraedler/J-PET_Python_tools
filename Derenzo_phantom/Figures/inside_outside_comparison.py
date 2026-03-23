"""
Compare the resolution inside and outside the scanner

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

# Auxiliary functions
from Derenzo_phantom.get_derenzo_contrast import get_derenzo_contrast_function
from Derenzo_phantom.get_derenzo_image import get_ground_truth_derenzo_image, load_derenzo_image
from Derenzo_phantom.psf_mtf_library import (fwhm_hermite_gaussian_definition, fwhm_hermite_gaussian_percentile, fwhm_hermite_gaussian_frequency,
                                             fwhm_plateau_polynomial_definition, fwhm_plateau_polynomial_percentile, fwhm_plateau_polynomial_frequency)
from Derenzo_phantom.psf_mtf_regression import fit_mtf
from Derenzo_phantom.print_significant_digits import print_value_and_error, error_propagation


def main():
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    derenzo_contrast_2d, derenzo_contrast_3d = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=True)

    it = 200
    img_all_true = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/ALL_true', it=it)
    img_tbtb_true = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TB-TB_true', it=it)
    img_tbbi_true = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TB-BI_true', it=it)
    img_bibi_true = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/BI-BI_true', it=it)

    img_all_energy = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/ALL_energy', it=it)
    img_tbtb_energy = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TB-TB_energy', it=it)
    img_tbbi_energy = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TB-BI_energy', it=it)
    img_bibi_energy = load_derenzo_image('/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/BI-BI_energy', it=it)

    img_tbtb_true_outside = load_derenzo_image('/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TB-TB_true', z_offset=755., it=it)
    img_tbtb_energy_outside = load_derenzo_image('/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TB-TB_energy', z_offset=755., it=it)

    images = [img_all_true, img_tbtb_true, img_tbbi_true, img_bibi_true,
              img_tbtb_true_outside]
    # images = [img_all_energy, img_tbtb_energy, img_tbbi_energy, img_bibi_energy,
    #           img_tbtb_energy_outside]

    wave_number_samples = np.linspace(0, 4, 1000)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    colors = colors[:4] + colors[1:2]
    line_styles = ['-'] * 4 + ['--']

    # dim = 1
    dim = 2

    for ii in range(len(images)):
        wave_numbers, contrast_values, contrast_errors = derenzo_contrast_2d(np.mean(images[ii], axis=-1))
        # wave_numbers, contrast_values, contrast_errors = derenzo_contrast_3d(images[ii])
        ax.errorbar(wave_numbers, contrast_values, yerr=contrast_errors, fmt='.', capsize=3, color=colors[ii], alpha=0.25)
        # ax.errorbar(wave_numbers, contrast_values, yerr=contrast_errors, fmt='.', capsize=3, color=colors[ii], alpha=0.0625)

        fitted_function, p_opt, p_err = fit_mtf(wave_numbers, contrast_values, contrast_errors, model='hermite-gaussian')
        # print_value_and_error(fwhm_hermite_gaussian(*p_opt, dim=dim), error_propagation(lambda sigma, alpha: fwhm_hermite_gaussian(sigma, alpha, dim=dim), p_opt, p_err))
        print_value_and_error(fwhm_hermite_gaussian_percentile(*p_opt, dim=dim), error_propagation(lambda sigma, alpha: fwhm_hermite_gaussian_percentile(sigma, alpha, dim=dim), p_opt, p_err))
        # print_value_and_error(fwhm_hermite_gaussian_mtf_50(*p_opt), error_propagation(fwhm_hermite_gaussian_mtf_50, p_opt, p_err))

        # fitted_function, p_opt, p_err = fit_mtf(wave_numbers, contrast_values, contrast_errors, model='plateau-polynomial')
        # print_value_and_error(fwhm_plateau_polynomial_definition(*p_opt), error_propagation(lambda k_one_half, alpha: fwhm_plateau_polynomial_definition(k_one_half, alpha, dim=dim), p_opt, p_err))
        # print_value_and_error(fwhm_plateau_polynomial_percentile(*p_opt, dim=dim), error_propagation(lambda k_one_half, alpha: fwhm_plateau_polynomial_percentile(k_one_half, alpha, dim=dim), p_opt, p_err))
        # print_value_and_error(fwhm_plateau_polynomial_frequency(*p_opt), error_propagation(fwhm_plateau_polynomial_frequency, p_opt, p_err))
        # print(p_opt)

        ax.plot(wave_number_samples, fitted_function(wave_number_samples), color=colors[ii], linestyle=line_styles[ii])
        # ax.plot(wave_number_samples, fitted_function(wave_number_samples), color=colors[ii], linestyle=line_styles[ii], alpha=0.25)

    # Dummy plots for the legend
    ax.plot(np.nan, color=colors[0], label='ALL')
    ax.plot(np.nan, color=colors[1], label='TB-TB')
    ax.plot(np.nan, color=colors[2], label='TB-BI')
    ax.plot(np.nan, color=colors[3], label='BI-BI')
    p1, = ax.plot(np.nan, linestyle='-', color='k')
    p2, = ax.plot(np.nan, linestyle='--', color='k')

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1.])
    ax.set_xlabel(r'Wave number $k$ [1/mm]')
    ax.set_ylabel('MTF')
    legend = ax.legend(frameon=False)
    ax.legend([p1, p2], ['Inside', 'Outside'], loc='lower left', frameon=False)
    ax.add_artist(legend)

    plt.show()

    return 0


if __name__ == "__main__":
    main()