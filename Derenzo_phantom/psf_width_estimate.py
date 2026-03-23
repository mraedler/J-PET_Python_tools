"""
Considering different width estimates for the PSF

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

# Auxiliary functions
from Derenzo_phantom.psf_mtf_library import (fwhm_hermite_gaussian_definition, fwhm_hermite_gaussian_percentile, fwhm_hermite_gaussian_frequency,
                                             fwhm_plateau_polynomial_definition, fwhm_plateau_polynomial_percentile, fwhm_plateau_polynomial_frequency)

def main():
    sigma = 2
    k_one_half = 1
    # k_one_half = 2

    alpha_samples = np.linspace(0, 1, 1001)

    p_hermite_gaussian_mtf = [1.60176445, 1.]
    p_hermite_gaussian_psf = [1.11793614, .192093153]

    p_plateau_polynomial_mtf = [1.14565584, 0.65155488]
    p_plateau_polynomial_psf = [1.17877477, 1.]

    # dim = 1
    dim = 2

    print(fwhm_hermite_gaussian_definition(*p_hermite_gaussian_mtf, dim=dim))
    print(fwhm_hermite_gaussian_definition(*p_hermite_gaussian_psf, dim=dim))

    print()

    print(fwhm_hermite_gaussian_percentile(*p_hermite_gaussian_mtf, dim=dim))
    print(fwhm_hermite_gaussian_percentile(*p_hermite_gaussian_psf, dim=dim))

    print()

    print(fwhm_hermite_gaussian_frequency(*p_hermite_gaussian_mtf))
    print(fwhm_hermite_gaussian_frequency(*p_hermite_gaussian_psf))

    print()
    print()

    print(fwhm_plateau_polynomial_definition(*p_plateau_polynomial_mtf, dim=dim))
    print(fwhm_plateau_polynomial_definition(*p_plateau_polynomial_psf, dim=dim))

    print()

    print(fwhm_plateau_polynomial_percentile(*p_plateau_polynomial_mtf, dim=dim))
    print(fwhm_plateau_polynomial_percentile(*p_plateau_polynomial_psf, dim=dim))

    print()

    print(fwhm_plateau_polynomial_frequency(*p_plateau_polynomial_mtf))
    print(fwhm_plateau_polynomial_frequency(*p_plateau_polynomial_psf))

    y_lim = [2.2, 4.8]

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.plot(alpha_samples, fwhm_hermite_gaussian_definition(sigma, alpha_samples, dim=1), color='tab:blue', linestyle='--')
    ax0.plot(alpha_samples, fwhm_hermite_gaussian_definition(sigma, alpha_samples, dim=2), color='tab:blue')
    ax0.plot(alpha_samples, fwhm_hermite_gaussian_percentile(sigma, alpha_samples, dim=1), color='tab:orange', linestyle='--')
    ax0.plot(alpha_samples, fwhm_hermite_gaussian_percentile(sigma, alpha_samples, dim=2), color='tab:orange')


    # beta = 1 / alpha_samples - 0.245
    # ll = 2 * np.sqrt(2) * np.sqrt(beta - lambertw(beta * np.exp(beta) / 2, k=0))
    # ax0.plot(alpha_samples, ll, color='red', linestyle='--')
    # ax0.plot(alpha_samples, fwhm_hermite_gaussian_percentile(sigma, alpha_samples, dim=1) - ll, color='tab:orange',
    #          linestyle='--')

    ax0.plot(alpha_samples, fwhm_hermite_gaussian_frequency(sigma, alpha_samples), color='tab:green')
    # ax0.set_ylim(2.8, 4.8)
    ax0.set_xlim(0, 1)
    ax0.set_xticklabels([])
    ax0.set_ylabel('FWHM [mm]')
    ax0.set_title('Hermite-Gaussian', fontsize=16)

    # Dummy plots for the legends
    p0, = ax0.plot(np.nan)
    p1, = ax0.plot(np.nan)
    p2, = ax0.plot(np.nan)
    legend = ax0.legend([p0, p1, p2], ['def', 'pct', 'frq'], frameon=False, loc='lower left')
    ax0.plot(np.nan, color='k', linestyle='-', label='1D')
    ax0.plot(np.nan, color='k', linestyle='--', label='2D')
    ax0.legend(loc='upper center', frameon=False, ncol=2)
    ax0.add_artist(legend)
    ax0.text(0.99, 4.4, r'$\sigma=2$ mm', ha='right', bbox={'facecolor':'none'})

    ax1.plot(alpha_samples, fwhm_plateau_polynomial_definition(k_one_half, alpha_samples, dim=1), color='tab:blue', linestyle='--')
    ax1.plot(alpha_samples, fwhm_plateau_polynomial_definition(k_one_half, alpha_samples, dim=2), color='tab:blue')
    ax1.plot(alpha_samples, fwhm_plateau_polynomial_percentile(k_one_half, alpha_samples, dim=1), color='tab:orange', linestyle='--')
    ax1.plot(alpha_samples, fwhm_plateau_polynomial_percentile(k_one_half, alpha_samples, dim=2), color='tab:orange')
    ax1.plot(alpha_samples, fwhm_plateau_polynomial_frequency(k_one_half, alpha_samples), color='tab:green')
    ax1.set_xlim(0, 1)
    # ax1.set_ylim(2.2, 4.6)

    ax1.set_title('Plateau-polynomial', fontsize=16)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('FWHM [mm]')
    ax1.text(0.99, 4, r'$k_{1/2}=1$ mm$^{-1}$', ha='right', bbox={'facecolor':'none'})

    plt.show()

    return 0


if __name__ == "__main__":
    main()
