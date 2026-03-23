"""
Point spread functions (PSFs) in one and two dimensions together with their modulation transfer functions (MTFs)

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from os.path import dirname
from scipy.optimize import root_scalar
from scipy.special import erf, lambertw, jv, struve
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
from tqdm import tqdm
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


def fwhm_hermite_gaussian_definition(sigma, alpha, dim=1):
    if dim not in [1, 2]:
        raise ValueError(f"Invalid dim={dim}. Expected 1 or 2.")

    alpha_threshold = 0.01
    alpha = np.array(alpha)
    beta_above = dim / 2 + 1 / alpha[alpha > alpha_threshold]
    alpha_below = alpha[alpha <= alpha_threshold]

    fwhm = np.zeros(alpha.shape)
    lambertw_values = lambertw(beta_above * np.exp(beta_above) / 2, k=0)
    if np.any(lambertw_values.imag != 0):
        print('Warning: nonzero imaginary part in the Lambert W function:')
        print(lambertw_values.imag)
    lambertw_values = lambertw_values.real
    fwhm[alpha > alpha_threshold] = 2 * np.sqrt(2) * np.sqrt(beta_above - lambertw_values) * sigma
    fwhm[alpha <= alpha_threshold] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt((2 - alpha_below) / (2 + alpha_below)) * sigma

    return fwhm


def get_hermite_gaussian_fwhm_percentile_samples(alpha_samples):
    x_samples = np.linspace(0, 10, 10000)

    x_mesh, alpha_mesh = np.meshgrid(x_samples, alpha_samples, indexing='ij')
    psf_1d = psf_hermite_gaussian_1d(x_mesh, 1, alpha_mesh)
    hwhm_1d = get_percentile(x_samples, alpha_samples, psf_1d, erf(np.sqrt(np.log(2))) / 2)
    fwhm_1d = 2 * hwhm_1d

    psf_2d = psf_hermite_gaussian_2d(x_mesh, 0, 1, alpha_mesh) * x_mesh * 2 * np.pi
    # hwhm_2d = get_percentile(x_samples, alpha_samples, psf_2d, 1 / 2)  # 50
    hwhm_2d = get_percentile(x_samples, alpha_samples, psf_2d, erf(np.sqrt(np.log(2))))  # 76
    fwhm_2d = 2 * hwhm_2d

    fig, ax = plt.subplots()
    ax.plot(alpha_samples, fwhm_1d)
    ax.plot(alpha_samples, fwhm_2d)
    ax.plot([alpha_samples[0], alpha_samples[-1]], [2 * np.sqrt(2 * np.log(2)), 2 * np.sqrt(2 * np.log(2))])
    plt.show()

    # np.save(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_76_1D.npy', fwhm_1d)
    # np.save(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_50_2D.npy', fwhm_2d)
    # np.save(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_76_2D.npy', fwhm_2d)

    return 0


def fwhm_hermite_gaussian_percentile(sigma, alpha, dim=1):
    alpha_samples = np.load(dirname(__file__) + '/FWHM_samples/alpha_samples.npy')

    if dim == 1:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_76_1D.npy')
    elif dim == 2:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_50_2D.npy')
        # fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/hermite_gaussian_percentile_76_2D.npy')
    else:
        raise ValueError(f"Invalid dim={dim}. Expected 1 or 2.")
    return np.interp(alpha, alpha_samples, fwhm_samples) * sigma


def fwhm_hermite_gaussian_frequency(sigma, alpha):
    alpha_threshold = 0.01
    alpha = np.array(alpha)
    beta_above = - 1 / alpha[alpha > alpha_threshold]
    alpha_below = alpha[alpha <= alpha_threshold]

    fwhm = np.zeros(alpha.shape)

    lambertw_values = lambertw(beta_above * np.exp(beta_above) / 2, k=-1)
    if np.any(lambertw_values.imag != 0):
        print('Warning: nonzero imaginary part in the Lambert W function:')
        print(lambertw_values.imag)
    lambertw_values = lambertw_values.real
    fwhm[alpha > alpha_threshold] = 4 * np.log(2) / (np.sqrt(2) * np.sqrt(beta_above - lambertw_values) / sigma)
    fwhm[alpha <= alpha_threshold] = 2 * np.sqrt(2 * np.log(2)) / np.sqrt(1 + alpha_below) * sigma
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


def get_plateau_polynomial_fwhm_samples(alpha_samples):
    hwhm_1d = np.zeros(alpha_samples.shape)
    hwhm_2d = np.zeros(alpha_samples.shape)

    tol = 1e-12

    for ii in range(alpha_samples.size):
        def root_function(xi): return psf_plateau_polynomial_1d(xi, 1., alpha_samples[ii]) - 1 / (2 * np.pi)
        res = root_scalar(root_function, x0=2.)
        hwhm_1d[ii] = res.root
        if abs(root_function(res.root)) > tol:
            print(root_function(res.root))

        def root_function(xi): return psf_plateau_polynomial_2d(xi, 0, 1., alpha_samples[ii]) - (1 + alpha_samples[ii] ** 2 / 5) / (4 * np.pi) / 2
        res = root_scalar(root_function, x0=2., x1=3.)
        hwhm_2d[ii] = res.root
        if abs(root_function(res.root)) > tol:
            print(root_function(res.root))

    fwhm_1d = 2 * hwhm_1d
    fwhm_2d = 2 * hwhm_2d

    fig, ax = plt.subplots()
    ax.plot(alpha_samples, fwhm_1d)
    ax.plot(alpha_samples, fwhm_2d)
    ax.plot([alpha_samples[0], alpha_samples[-1]], [4.430178735448465, 4.430178735448465])  # Mathematica result
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'FWHM')
    plt.show()

    np.save(dirname(__file__) + '/FWHM_samples/plateau_polynomial_1D.npy', fwhm_1d)
    np.save(dirname(__file__) + '/FWHM_samples/plateau_polynomial_2D.npy', fwhm_2d)

    return 0


def fwhm_plateau_polynomial_definition(k_one_half, alpha, dim=1):
    alpha_samples = np.load(dirname(__file__) + '/FWHM_samples/alpha_samples.npy')

    if dim == 1:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/plateau_polynomial_1D.npy')
    elif dim == 2:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/plateau_polynomial_2D.npy')
    else:
        raise ValueError(f"Invalid dim={dim}. Expected 1 or 2.")
    return np.interp(alpha, alpha_samples, fwhm_samples) / k_one_half


def get_plateau_polynomial_fwhm_percentile_samples(alpha_samples):
    x_samples = np.linspace(0, 10, 10000)

    x_mesh, alpha_mesh = np.meshgrid(x_samples, alpha_samples, indexing='ij')
    # psf_1d = psf_plateau_polynomial_1d(x_mesh, 1, alpha_mesh)
    psf_1d = np.stack([psf_plateau_polynomial_1d(x_samples, 1, alpha) for alpha in alpha_samples], axis=1)
    hwhm_1d = get_percentile(x_samples, alpha_samples, psf_1d, erf(np.sqrt(np.log(2))) / 2, plot_cdf=False)
    fwhm_1d = 2 * hwhm_1d

    # psf_2d = psf_hermite_gaussian_2d(x_mesh, 0, 1, alpha_mesh) * x_mesh * 2 * np.pi
    psf_2d = np.stack([psf_plateau_polynomial_2d(x_samples, 0, 1, alpha) * x_samples * 2 * np.pi for alpha in tqdm(alpha_samples)], axis=1)
    # hwhm_2d = get_percentile(x_samples, alpha_samples, psf_2d, 1 / 2, plot_cdf=True)  # 50
    hwhm_2d = get_percentile(x_samples, alpha_samples, psf_2d, erf(np.sqrt(np.log(2))), plot_cdf=True)  # 76
    fwhm_2d = 2 * hwhm_2d

    fig, ax = plt.subplots()
    ax.plot(alpha_samples, fwhm_1d)
    ax.plot(alpha_samples, fwhm_2d)
    # ax.plot([alpha_samples[0], alpha_samples[-1]], [2 * np.sqrt(2 * np.log(2)), 2 * np.sqrt(2 * np.log(2))])
    plt.show()

    # np.save(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_76_1D.npy', fwhm_1d)
    # np.save(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_50_2D.npy', fwhm_2d)
    np.save(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_76_2D.npy', fwhm_2d)

    return 0


def fwhm_plateau_polynomial_percentile(k_one_half, alpha, dim=1):
    alpha_samples = np.load(dirname(__file__) + '/FWHM_samples/alpha_samples.npy')

    if dim == 1:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_76_1D.npy')
    elif dim == 2:
        fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_50_2D.npy')
        # fwhm_samples = np.load(dirname(__file__) + '/FWHM_samples/plateau_polynomial_percentile_76_2D.npy')
    else:
        raise ValueError(f"Invalid dim={dim}. Expected 1 or 2.")
    return np.interp(alpha, alpha_samples, fwhm_samples) / k_one_half


def fwhm_plateau_polynomial_frequency(k_one_half, alpha):
    alpha = np.array(alpha)
    return 4 * np.log(2) / k_one_half * np.ones(alpha.shape)


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


""""""


def get_percentile(x, y, psf, p, plot_cdf=False):
    # cdf = cumulative_trapezoid(psf, x=x, axis=0, initial=0)
    cdf = cumulative_simpson(psf, x=x, axis=0, initial=0)
    x_idx, alpha_idx = np.where((cdf[:-1, :] - p) * (cdf[1:, :] - p) < 0)

    # if alpha_idx.size > psf.shape[1]:
    #     fig, ax = plt.subplots()
    #     ax.plot(y[alpha_idx], x[x_idx])
    #     plt.show()
    #
    #     alpha_idx = alpha_idx[:psf.shape[1]]
    #     x_idx = x_idx[:psf.shape[1]]

    if plot_cdf:
        fig, ax = plt.subplots()
        ax.imshow(cdf.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
        ax.contour(x, y, cdf.T, [p])
        ax.set_aspect('auto')
        plt.show()

    idx_sort = np.argsort(alpha_idx)
    alpha_idx = alpha_idx[idx_sort]
    x_idx = x_idx[idx_sort]

    x_0, x_1 = x[x_idx], x[x_idx + 1]
    y_0, y_1 = cdf[x_idx, alpha_idx], cdf[x_idx + 1, alpha_idx]

    hwhm = x_0 + (p - y_0) / (y_1 - y_0) * (x_1 - x_0)
    return hwhm


if __name__ == '__main__':
    # visualize_hermite_gaussian()
    # visualize_plateau_polynomial()

    # alpha_samples = np.linspace(0, 1, 1001)
    # np.save(dirname(__file__) + '/FWHM_samples/alpha_samples.npy', alpha_samples)
    alpha_samples = np.load(dirname(__file__) + '/FWHM_samples/alpha_samples.npy')

    # get_plateau_polynomial_fwhm_samples(alpha_samples)
    # get_hermite_gaussian_fwhm_percentile_samples(alpha_samples)
    # get_plateau_polynomial_fwhm_percentile_samples(alpha_samples)
    pass
