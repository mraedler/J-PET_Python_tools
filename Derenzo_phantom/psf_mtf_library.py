"""
Point spread functions (PSFs) in one and two dimensions together with their modulation transfer functions (MTFs)

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import lambertw, jv, struve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize


def get_mtf(model='gaussian'):

    if model == 'gaussian':
        def mtf_model(k, sigma):
            return mtf_hermite_gaussian(k, sigma, 0.)

        p0 = [1.]
        lower_bounds = [0.]
        upper_bounds = [np.inf]

    elif model == 'hermite-gaussian':
        def mtf_model(k, sigma, alpha):
            return mtf_hermite_gaussian(k, sigma, alpha)

        p0 = [1., 0.5]
        lower_bounds = [0., 0.]
        upper_bounds = [np.inf, 1.]

    elif model == 'lorentzian-raised':
        def mtf_model(k, k_one_half, n):
            return 1 / (1 + (k / k_one_half) ** 2) ** n

        p0 = [1., 2.]
        lower_bounds = [0., 1.]
        upper_bounds = [np.inf, 20.]

    elif model == 'lorentzian-generalized':
        def mtf_model(k, k_one_half, n):
            return 1 / (1 + (k / k_one_half) ** (2 * n))

        p0 = [1., 2.]
        lower_bounds = [0., 1.]
        upper_bounds = [np.inf, 10.]

    elif model == 'plateau-polynomial':
        def mtf_model(k, k_one_half, alpha):
            return mtf_plateau_polynomial(k, k_one_half, alpha)

        p0 = [1., 0.5]
        lower_bounds = [0., 0.]
        upper_bounds = [np.inf, 1.]

    elif model == 'plateau-sine':
        def mtf_model(k, k_one_half, alpha):
            return mtf_plateau_sine(k, k_one_half, alpha)

        p0 = [1., 0.5]
        lower_bounds = [0., 0.]
        upper_bounds = [np.inf, 1.]

    elif model == 'plateau-gaussian':

        def mtf_model(k, k_one_half, alpha):
            return mtf_plateau_gaussian(k, k_one_half, alpha)

        p0 = [1., 0.5]
        lower_bounds = [0., 0.]
        upper_bounds = [np.inf, 1.]

    else:
        sys.exit("Error: unknown model '%s'." % model)

    return p0, lower_bounds, upper_bounds, mtf_model


"""Hermite-Gaussian PSF and MTF"""


def mtf_hermite_gaussian(k, sigma, alpha):
    argument = k ** 2 * sigma ** 2 / 2
    return np.exp(- argument) * (1 - alpha * (- argument))


def psf_hermite_gaussian_1d(x, sigma, alpha):
    argument = x ** 2 / (2 * sigma ** 2)
    gaussian = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-argument)
    return gaussian * (1 - alpha * (argument - 1 / 2))


def psf_hermite_gaussian_2d(x, y, sigma, alpha):
    argument = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    gaussian = 1 / (2 * np.pi * sigma ** 2) * np.exp(-argument)
    return gaussian * (1 - alpha * (argument - 1 / 1))


def fwhm_hermite_gaussian_1d(sigma, alpha):
    alpha_threshold = 0.01
    alpha = np.array(alpha)
    beta_above = 1 / 2 + 1 / alpha[alpha > alpha_threshold]
    alpha_below = alpha[alpha <= alpha_threshold]

    fwhm = np.zeros(alpha.shape)
    lambertw_values = lambertw(beta_above * np.exp(beta_above) / 2, k=0)
    if np.any(lambertw_values.imag != 0):
        print('Warning: nonzero imaginary part in the Lambert W function:')
        print(lambertw_values.imag)
    lambertw_values = lambertw_values.real
    fwhm[alpha > alpha_threshold] = 2 * np.sqrt(2) * np.sqrt(beta_above - lambertw_values) * sigma
    fwhm[alpha <= alpha_threshold] = 2 * np.sqrt(2) * np.sqrt(np.log(2)) * np.sqrt((2 - alpha_below) / (2 + alpha_below)) * sigma

    return fwhm


"""Plateau-polynomial PSF and MTF"""


def mtf_plateau_polynomial(k, k_one_half, alpha):
    if alpha != 0.:
        x = (k - k_one_half) / (alpha * k_one_half)
    else:
        x = np.sign(k - k_one_half) * np.inf
    y = np.ones_like(x)
    drop_region = (x > -1) & (x < 1)
    y[drop_region] = (2 - 3 * x[drop_region] + x[drop_region] ** 3) / 4
    y[x >= 1] = 0
    return y


def psf_plateau_polynomial_1d(x, k_one_half, alpha):
    xi = k_one_half * x
    y = np.zeros_like(x)

    y[xi == 0.] = k_one_half / np.pi
    xi_not_zero = xi != 0.

    xi = xi[xi_not_zero]
    y[xi_not_zero] = k_one_half / np.pi * np.sin(xi) / xi

    if alpha != 0.:
        y[xi_not_zero] *= (np.sin(alpha * xi) - alpha * xi * np.cos(alpha * xi)) / (alpha ** 3 * xi ** 3 / 3)

    return y


def psf_plateau_polynomial_2d(x, y, k_one_half, alpha):
    rho = np.sqrt(x ** 2 + y ** 2)
    psf = np.zeros_like(rho)

    rho_is_zero = rho == 0.
    psf[rho_is_zero] = k_one_half ** 2 * (1 + alpha ** 2 / 5) / (4 * np.pi)
    rho = rho[~rho_is_zero]

    if alpha == 0.:
        psf[~rho_is_zero] = 1 / (2 * np.pi) * k_one_half * jv(1, k_one_half * rho) / rho
    else:
        km = (1 - alpha) * k_one_half
        kp = (1 + alpha) * k_one_half
        psf_2d_one = (1 - alpha ** 2) * (bessel_integral_f(kp * rho) - bessel_integral_f(km * rho))
        psf_2d_two = 3 / (k_one_half ** 2 * rho ** 2) * (bessel_integral_g(kp * rho) - bessel_integral_g(km * rho))
        psf[~rho_is_zero] = 1 / (2 * np.pi) * 3 / 4 * 1 / (alpha ** 3 * rho ** 2) * 1 / (k_one_half * rho) * (psf_2d_one - psf_2d_two)

    return psf


def bessel_integral_f(x):
    return bessel_integral_h(x) + x * jv(2, x)


def bessel_integral_g(x):
    # return np.pi * x * (jv(1, x) * struve(2, x) - jv(2, x) * struve(1, x)) / 2
    return bessel_integral_h(x) + x ** 2 * jv(1, x) / 3


def bessel_integral_h(x):
    return np.pi * x * (jv(0, x) * struve(1, x) - jv(1, x) * struve(0, x)) / 2


def get_plateau_polynomial_xi_one_half_samples():
    alpha = np.linspace(0, 1, 1001)
    xi_one_half = np.zeros(alpha.shape)

    for ii in range(alpha.size):
        def root_function(xi): return psf_plateau_polynomial_1d(xi, 1., alpha[ii]) - 1 / (2 * np.pi)
        res = root_scalar(root_function, x0=2.)
        xi_one_half[ii] = res.root
        # print(root_function(res.root))

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(alpha, xi_one_half)
    ax.set_xlim(0, 1)
    ax.set_ylim([1.48, 1.92])
    ax.set_yticks([1.6, 1.8])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\xi_{1/2}$')
    plt.show()

    # np.save(sys.path[0] + '/FWHM/alpha.npy', alpha)
    # np.save(sys.path[0] + '/FWHM/xi_one_half.npy', xi_one_half)

    return 0


def fwhm_plateau_polynomial_1d(k_one_half, alpha):
    # alpha_samples = np.load(sys.path[0] + '/FWHM/alpha.npy')
    # xi_one_half_samples = np.load(sys.path[0] + '/FWHM/xi_one_half.npy')
    alpha_samples = np.load('/home/martin/PycharmProjects/J-PET_Python_tools/Derenzo_phantom/FWHM/alpha.npy')
    xi_one_half_samples = np.load('/home/martin/PycharmProjects/J-PET_Python_tools/Derenzo_phantom/FWHM/xi_one_half.npy')
    return 2 * np.interp(alpha, alpha_samples, xi_one_half_samples) / k_one_half


"""Other plateau MTFs"""


def mtf_plateau_sine(k, k_one_half, alpha):
    if alpha != 0.:
        x = (k - k_one_half) / (alpha * k_one_half)
    else:
        x = np.sign(k - k_one_half) * np.inf
    y = np.ones_like(x)
    drop_region = (x > -1) & (x < 1)
    y[drop_region] = (1 - np.sin(np.pi / 2 * x[drop_region])) / 2
    y[x >= 1] = 0
    return y


def mtf_plateau_gaussian(k, k_one_half, alpha):
    if alpha != 0.:
        x = (k - k_one_half) / (alpha * k_one_half)
    else:
        x = np.sign(k - k_one_half) * np.inf
    y = np.ones_like(x)
    drop_region = (x > -1)
    y[drop_region] = 2 ** (- (1 + x[drop_region]) ** 2)
    return y


"""Visualization of one dimensional PSFs/MTFs"""


def visualize_hermite_gaussian():
    sigma = 1

    x = np.linspace(-4 * sigma, 4 * sigma, 200 + 1)
    k = np.linspace(0, 4 / sigma, 100)

    n = 11
    cmap = plt.get_cmap('viridis', n)
    alpha_edges = (np.arange(n + 1) - 1 / 2) / (n - 1)
    alpha = (alpha_edges[1:] + alpha_edges[:-1]) / 2

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.5))
    for ii in range(alpha.size):
        ax0.plot(x, psf_hermite_gaussian_1d(x, sigma, alpha[ii]), color=cmap(ii))
        ax1.plot(k, mtf_hermite_gaussian(k, sigma, alpha[ii]), color=cmap(ii))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm, cax=cax, ticks=alpha[::2], boundaries=alpha_edges, label=r'$\alpha$')

    # ax0.text(2, 0.5, r'$\sigma = 1$ mm')
    ax0.set_xlim(x[0], x[-1])
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_title(r'$\mathrm{PSF}(x)$')

    # ax1.text(3, 0.75, r'$\sigma = 1$ mm')
    ax1.set_xlim(k[0], k[-1])
    ax1.set_xlabel(r'$k$ [1/mm]')
    ax1.set_xticks(np.arange(5))
    ax1.set_title(r'$\mathrm{MTF}(k)$')
    plt.show()

    return 0


def visualize_plateau_polynomial():
    k_one_half = 2

    x = np.linspace(-16 / k_one_half, 16 / k_one_half, 200 + 1)
    k = np.linspace(0, 2 * k_one_half, 100)

    n = 11
    cmap = plt.get_cmap('viridis', n)
    alpha_edges = (np.arange(n + 1) - 1 / 2) / (n - 1)
    alpha = (alpha_edges[1:] + alpha_edges[:-1]) / 2

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.5))
    for ii in range(alpha.size):
        ax0.plot(x, psf_plateau_polynomial_1d(x, k_one_half, alpha[ii]), color=cmap(ii))
        ax1.plot(k, mtf_plateau_polynomial(k, k_one_half, alpha[ii]), color=cmap(ii))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm, cax=cax, ticks=alpha[::2], boundaries=alpha_edges, label=r'$\alpha$')

    ax0.set_xlim(x[0], x[-1])
    ax0.set_xticks([-8, -4, 0, 4, 8])
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_title(r'$\mathrm{PSF}(x)$')

    ax1.set_xlim(k[0], k[-1])
    ax1.set_xlabel(r'$k$ [1/mm]')
    ax1.set_xticks(np.arange(5))
    ax1.set_title(r'$\mathrm{MTF}(k)$')
    plt.show()

    return 0


if __name__ == '__main__':
    # visualize_hermite_gaussian()
    # visualize_plateau_polynomial()
    # get_plateau_polynomial_xi_one_half_samples()
    pass
