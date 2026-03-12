"""
Point spread functions (PSFs) in one and two dimensions together with their modulation transfer functions (MTFs)

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit, root_scalar
from scipy.special import lambertw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize


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

    return 0


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

    return 0


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
    alpha_samples = np.load(sys.path[0] + '/FWHM/alpha.npy')
    xi_one_half_samples = np.load(sys.path[0] + '/FWHM/xi_one_half.npy')
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


"""Convenience functions for printing resulting fit parameters"""


def finite_differences(function, inputs, delta=0.001):

    derivatives = np.zeros(inputs.size)

    for ii in range(inputs.size):
        inputs_forward = inputs + np.eye(1, inputs.size, k=ii).flatten() * delta
        inputs_backward = inputs - np.eye(1, inputs.size, k=ii).flatten() * delta

        derivatives[ii] = (function(*inputs_forward) - function(*inputs_backward)) / (2 * delta)

    return derivatives


def error_propagation(function, inputs, inputs_error):

    derivatives = finite_differences(function, inputs)
    gaussian_error = np.sqrt(np.sum(derivatives ** 2 * inputs_error ** 2))

    return gaussian_error


def round_to_exponent(value, exponent):
    return np.round(value / 10 ** exponent) * 10 ** exponent


def print_value_and_error(value, error):
    exp_error = np.floor(np.log10(np.abs(error)))
    error_rounded = round_to_exponent(error, exp_error)
    exp_error_rounded = np.floor(np.log10(np.abs(error_rounded)))
    error_rounded = round_to_exponent(error_rounded, exp_error_rounded)
    value_rounded = round_to_exponent(value, exp_error_rounded)

    format_specifier = '%1.5f ± %1.5f  ⇒  %1.' + ('%s' % int(abs(exp_error_rounded))) + 'f(%d)'
    print(format_specifier % (value, error, value_rounded, int(error_rounded / 10 ** exp_error_rounded)))

    return 0













""""""


def alpha_beta_range():

    alpha_edges = np.linspace(-3, 3, 201)
    beta_edges = np.linspace(-1, 3, 203)

    alpha_centers = (alpha_edges[1:] + alpha_edges[:-1]) / 2
    beta_centers = (beta_edges[1:] + beta_edges[:-1]) / 2

    print(beta_centers)

    alpha_mesh, beta_mesh = np.meshgrid(alpha_centers.astype(np.complex128), beta_centers.astype(np.complex128), indexing='ij')

    x0 = x_0_zero_crossing(alpha_mesh, beta_mesh)
    x1 = x_1_zero_crossing(alpha_mesh, beta_mesh)
    x2 = x_0_zero_crossing2(alpha_mesh, beta_mesh)


    x0_not_imaginary = (np.imag(x0) != 0)
    x1_not_imaginary = (np.imag(x1) != 0)
    x2_not_imaginary = (np.imag(x2) != 0)

    aa = (x0_not_imaginary) & (x1_not_imaginary) & (x2_not_imaginary)


    fig, ax = plt.subplots()
    ax.imshow(aa.T, origin='lower', extent=(alpha_edges[0], alpha_edges[-1], beta_edges[0], beta_edges[-1]))
    plt.show()

    return 0


def x_0_zero_crossing(alpha_mesh, beta_mesh):
    x_0 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_0[~beta_0] = np.sqrt(-alpha_mesh[~beta_0] / beta_mesh[~beta_0] + np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0]) / beta_mesh[~beta_0])
    x_0[beta_0] = 1j / np.sqrt(alpha_mesh[beta_0])

    return x_0


def x_1_zero_crossing(alpha_mesh, beta_mesh):
    x_1 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_1[~beta_0] = np.sqrt(-alpha_mesh[~beta_0] / beta_mesh[~beta_0] - np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0]) / beta_mesh[~beta_0])
    x_1[beta_0] = 1j * np.inf

    return x_1


def x_0_zero_crossing2(alpha_mesh, beta_mesh):
    x_0 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_0[~beta_0] = np.sqrt(1 - alpha_mesh[~beta_0] / beta_mesh[~beta_0] + np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0] + beta_mesh[~beta_0] ** 2) / beta_mesh[~beta_0])
    # x_0[beta_0] = 1j / np.sqrt(alpha_mesh[beta_0])

    return x_0








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
    alpha_beta_range()
