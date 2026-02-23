"""
Point spread function

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize


def psf_image_domain_1d(x, sigma, alpha):
    argument = -x ** 2 / (2 * sigma ** 2)
    gaussian = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(argument)
    return gaussian * (1 - alpha * (- 1 / 2 - argument))


def psf_frequency_domain_1d(k, sigma, alpha):
    argument = - k ** 2 * sigma ** 2 / 2
    return np.exp(argument) * (1 - alpha * argument)


def psf_image_domain_2d():

    return 0


def fit_psf_frequency_domain_1d(frequencies, contrast_values, contrast_errors, include_scaling=False):

    def fit_function(k, sigma, alpha):
        return psf_frequency_domain_1d(k, sigma, alpha)

    p0 = [1., 0.5]
    lower_bounds = [0., 0]
    upper_bounds = [np.inf, 1.]

    if include_scaling:
        def fit_function(k, sigma, alpha, scale):
            return psf_frequency_domain_1d(k, sigma, alpha) * scale

        p0.append(1.)
        lower_bounds.append(-1.)
        upper_bounds.append(1.)

    # # noinspection PyTupleAssignmentBalance
    # p_opt, p_cov = curve_fit(fit_function, frequencies, contrast_values,
    #                          p0=p0,
    #                          bounds=(lower_bounds,
    #                                  upper_bounds),
    #                          sigma=contrast_errors,
    #                          absolute_sigma=True,
    #                          maxfev=10000)

    # noinspection PyTupleAssignmentBalance
    p_opt, p_cov = curve_fit(fit_function, frequencies, contrast_values,
                             p0=p0,
                             bounds=(lower_bounds,
                                     upper_bounds),
                             maxfev=10000)

    # print(p_cov)
    p_err = np.sqrt(np.diag(p_cov))
    # p_err[p_err > 1e2] = 0

    return lambda k: fit_function(k, *p_opt), p_opt, p_err


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



def visualize_gaussian_expansion():
    sigma = 1

    x = np.linspace(-4 * sigma, 4 * sigma, 200 + 1)
    k = np.linspace(0, 4 / sigma, 100)

    N = 11
    cmap = plt.get_cmap('viridis', N)
    alpha_edges = (np.arange(N + 1) - 1 / 2) / (N - 1)
    alpha = (alpha_edges[1:] + alpha_edges[:-1]) / 2

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.5))
    for ii in range(alpha.size):
        ax0.plot(x, psf_image_domain_1d(x, sigma, alpha[ii]), color=cmap(ii))
        ax1.plot(k, psf_frequency_domain_1d(k, sigma, alpha[ii]), color=cmap(ii))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm, cax=cax, ticks=alpha[::2], boundaries=alpha_edges, label=r'$\alpha$')

    # ax0.text(2, 0.5, r'$\sigma = 1$ mm')
    ax0.set_xlim(x[0], x[-1])
    ax0.set_xlabel(r'$x$ [mm]')
    # ax0.set_title(r'Spatial domain $\psi(x)$')
    ax0.set_title(r'$\mathrm{PSF}(x)$')

    # ax1.text(3, 0.75, r'$\sigma = 1$ mm')
    ax1.set_xlim(k[0], k[-1])
    ax1.set_xlabel(r'$k$ [1/mm]')
    ax1.set_xticks(np.arange(5))
    # ax1.set_title(r'Frequency domain $\tilde{\psi}(k)$')
    ax1.set_title(r'$\mathrm{MTF}(k)$')

    plt.show()

    return 0


if __name__ == '__main__':
    visualize_gaussian_expansion()
    # alpha_beta_range()
