"""
Set of point spread functions in 2D

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.special import erf, jv, lambertw
from scipy.special import gamma as gamma_function
# from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import make_lsq_spline, LSQUnivariateSpline, splrep, splev, interp1d
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_2d_point_spread_function(tag, return_fourier_profile=False):

    if tag == 'gaussian':
        point_spread_function = gaussian_2d
        # def fwhm(param): return 2 * np.sqrt(2 * np.log(2)) * param[0]
        def fwhm(param): return 2 * np.sqrt(2 * np.log(2)) * param

        p_0 = np.array([1.])
        p_s = np.array([1.])

    elif tag == 'lorentzian':
        point_spread_function = lorentzian_2d
        def fwhm(param): return 2 * np.sqrt(2 ** (2/3) - 1) * param[0]

        p_0 = np.array([1.])
        p_s = np.array([1.])

    elif tag == 'generalized lorentzian':
        point_spread_function = generalized_lorentzian_2d

        def fwhm(params):
            return 2 * (2 ** (params[1] / (1 + params[1])) - 1) ** (1 / params[1]) * params[0]

        p_0 = np.array([1., 2])
        p_s = np.array([1., 2])

    elif tag == 'generalized center lorentzian':
        point_spread_function = generalized_center_lorentzian_2d
        def fwhm(params): return 2 * params[0]

        p_0 = np.array([1., 3])
        p_s = np.array([1., 3])

    elif tag == 'polynomial':
        point_spread_function = polynomial_2d

        def fwhm(params, return_error=False):
            if return_error:
                return 0
            return 2 * (23/270000 * params[1] ** 2 + 23/3600 * params[1] + 1/2) * params[0]

        p_0 = np.array([1., -5])
        p_s = np.array([1., 5])

    elif tag == 'bspline':
        point_spread_function = bspline

        def fwhm(params):
            return 0

        k, n_rho = 1, 10
        p_0 = np.concatenate((np.ones(k), np.zeros(n_rho + k)))
        p_s = np.ones(p_0.shape)

    elif tag == 'interpolation':
        rho_0, rho_1 = 0, 10
        k, n_rho = 1, 12
        rho = np.linspace(rho_0, rho_1, n_rho)
        point_spread_function = lambda x_grid, y_grid, param: interpolation(x_grid, y_grid, param, rho)

        def fwhm(params):
            return 0

        p_0 = np.zeros(rho.shape, dtype=float)
        p_0[0] = 1.
        p_s = np.ones(rho.shape, dtype=float)

    elif tag == 'gaussian-expansion':
        point_spread_function = gaussian_expansion

        def fwhm(params):
            sigma = params[0]
            alpha = params[1]
            if alpha < 0.001:
                return 2 * np.sqrt(2 * np.log(2) * (1 - alpha)) * sigma
            if alpha >= 0.001:
                return 2 * np.sqrt(2 + 2 / alpha - 2 * lambertw(np.exp(1 + 1 / alpha) * (1 + alpha) / (2 * alpha)).real) * sigma

        # def fourier_profile(x, params):
        def fourier_profile(x, sigma_in, alpha):
            sigma = sigma_in * 2 * np.pi
            return (1 + alpha * x ** 2 * sigma ** 2 / 2) * np.exp(- x ** 2 * sigma ** 2 / 2)

        p_0 = np.array([1., 0.5])
        p_s = np.array([1., 0.5])
        # p_0 = np.array([1.7, 0.5, 0.5])
        # p_s = np.array([1., 0.25, 0.25])

    elif tag == 'bessel':
        point_spread_function = bessel_2d

        a6_data = np.load('FWHM/a6_bessel.npy')
        fwhm_bessel_data = np.load('FWHM/fwhm_bessel.npy')

        def fwhm(params):
            return params[0] * np.interp(params[1], a6_data, fwhm_bessel_data)

        def fourier_profile(x, delta_in, a6):
            delta = delta_in * 2 * np.pi
            return (1 - (delta * x) ** 2) ** 2 * (1 + a6 * (delta * x) ** 2) * np.heaviside(1 - (x * delta) ** 2, 0.5)

        p_0 = np.array([1., 0.5])
        p_s = np.array([1., 1])

    else:
        sys.exit('Unknown point spread function.')

    if return_fourier_profile:
        return point_spread_function, fwhm, p_0, p_s, fourier_profile

    return point_spread_function, fwhm, p_0, p_s


def gaussian_2d(x_grid, y_grid, param):
    # sigma_scaled = np.sqrt(2) * param[0]
    sigma_scaled = np.sqrt(2) * param

    x_itg = 1 / 2 * (erf(x_grid[:-1] / sigma_scaled) - erf(x_grid[1:] / sigma_scaled)).reshape((x_grid.size - 1, 1))
    y_itg = 1 / 2 * (erf(y_grid[:-1] / sigma_scaled) - erf(y_grid[1:] / sigma_scaled)).reshape((1, y_grid.size - 1))

    # Plot the PDF
    x_cp, _ = upscale_samples(x_grid, 3)
    # cp = 1 / np.sqrt(2 * np.pi * param ** 2) * np.exp(- x_cp ** 2 / (2 * param ** 2))
    cp = 1 / (2 * np.pi * param ** 2) * np.exp(- x_cp ** 2 / (2 * param ** 2))

    return x_itg * y_itg, x_cp, cp


def lorentzian_2d(x_grid, y_grid, param):
    # gamma = param[0]
    gamma = param

    x0_grid, y0_grid = np.meshgrid(x_grid[:-1], y_grid[:-1], indexing='ij')
    x1_grid, y1_grid = np.meshgrid(x_grid[1:], y_grid[1:], indexing='ij')

    def l_itg(x, y): return np.arctan(x * y / (gamma * np.sqrt(x ** 2 + y ** 2 + gamma ** 2)))
    itg_x_y = (l_itg(x0_grid, y0_grid) - l_itg(x1_grid, y0_grid) - l_itg(x0_grid, y1_grid) + l_itg(x1_grid, y1_grid)) / (2 * np.pi)

    # Plot the PDF
    x_cp, _ = upscale_samples(x_grid, 3)
    cp = 1 / (np.pi * gamma) * 1 / (x_cp ** 2 / gamma ** 2 + 1)

    return itg_x_y, x_cp, cp


def generalized_lorentzian_2d(x_grid, y_grid, params):
    gamma = params[0]
    n = params[1]

    norm = 1 / (2 * np.pi) * gamma_function(1 / n) / (gamma_function(2 / n) * gamma_function(1 - 1 / n)) * 1 / gamma ** 2
    def pdf(x, y): return norm / (np.sqrt((x / gamma) ** 2 + (y / gamma) ** 2) ** n + 1) ** (1 + 1 / n)

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)
    return itg_x_y, x_cp, cp


def generalized_center_lorentzian_2d(x_grid, y_grid, params):
    gamma = params[0]
    n = params[1]

    norm = 1 / np.pi * np.sin(2 * np.pi / n) / (2 * np.pi / n) * 1 / gamma ** 2
    def pdf(x, y): return norm / (np.sqrt((x / gamma) ** 2 + (y / gamma) ** 2) ** n + 1)

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)
    return itg_x_y, x_cp, cp


def polynomial_2d(x_grid, y_grid, params):
    delta = params[0]
    a4 = params[1]

    def pdf(x, y):
        rho_p = np.sqrt(x ** 2 + y ** 2) / delta
        a0 = - 1 / 9 * (a4 - 60)
        a1 = 0
        a2 = 4 / 3 * (a4 - 15)
        a3 = - 20 / 9 * (a4 - 6)
        poly = a0 + a1 * rho_p + a2 * rho_p ** 2 + a3 * rho_p ** 3 + a4 * rho_p ** 4
        return poly / (2 * np.pi * delta ** 2) * np.heaviside(1 - rho_p, 0.5)

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)
    return itg_x_y, x_cp, cp


def bspline(x_grid, y_grid, params, k=1, rho_0=0., rho_1=10., n_rho=10):
    d_rho = (rho_1 - rho_0) / (n_rho - 1)
    # t = rho_0 + np.arange(-3, n_rho + 3) * d_rho
    t = np.concatenate((rho_0 + np.arange(-k, 0), rho_0 + np.arange(n_rho) * d_rho, rho_1 + np.arange(1, k + 1)))
    rho = np.linspace(rho_0, rho_1, num=100)

    x = (x_grid[50:61] - 0.5)
    y = np.exp(-x ** 2 / (2 * 2 ** 2))
    def objective_function(cc):
        return np.sum((splev(x, (t, cc, k), ext=2) - y) ** 2)
    p_opt = minimize(objective_function, method='Nelder-Mead', x0=params)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'x-')
    ax.plot(rho, splev(rho, (t, params, k), ext=2))
    ax.plot(rho, splev(rho, (t, p_opt.x, k), ext=2))
    # ax.plot(t, p_opt.x, 'x')
    plt.show()

    norm = np.trapz(splev(rho, (t, params, k)) * rho, x=rho)

    def pdf(x, y):
        rho_p = np.sqrt(x ** 2 + y ** 2)
        return splev(rho_p, (t, params, k)) / (2 * np.pi * norm) * np.heaviside(rho_1 - rho_p, 0.5)

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)
    return itg_x_y, x_cp, cp


def interpolation(x_grid, y_grid, params, rho):

    # x = (x_grid[50:61] - 0.5)
    # y = np.exp(-x ** 2 / (2 * 2 ** 2))
    #
    # initial_simplex = np.tile(params, (params.size + 1, 1))
    # initial_simplex[1:, :] += np.diag(np.ones(params.shape))
    #
    # def objective_function(cc):
    #     return np.sum((interp1d(rho, cc, kind='linear')(x) - y) ** 2)
    # p_opt = minimize(objective_function, method='Nelder-Mead', x0=params, options={'initial_simplex': initial_simplex, 'maxfev': 10000})
    # print(p_opt)
    # # sys.exit()
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, y, 'x-')
    # ax.plot(rho, p_opt.x)
    # ax.plot(x, interp1d(rho, p_opt.x, kind='linear')(x), '--')
    # plt.show()

    rho_samples = np.linspace(rho[0], rho[-1], num=100)
    norm = np.trapz(interp1d(rho, params, kind='linear')(rho_samples) * rho_samples, x=rho_samples)

    def pdf(x, y):
        rho_p = np.sqrt(x ** 2 + y ** 2)
        return interp1d(rho, params, kind='linear', bounds_error=False, fill_value=0.)(rho_p) / (2 * np.pi * norm) * np.heaviside(rho[-1] - rho_p, 0.5)

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)
    # print(np.sum(itg_x_y))
    # fig, ax = plt.subplots()
    # ax.imshow(itg_x_y)
    # plt.show()

    return itg_x_y, x_cp, cp


def bessel_2d(x_grid, y_grid, params):
    delta = params[0]
    a6 = params[1]

    def pdf(x, y):
        out = np.zeros(x.shape)
        rho_p = np.sqrt(x ** 2 + y ** 2) / delta

        out[rho_p == 0] = (4 + a6) / (48 * np.pi * delta ** 2)

        crit = rho_p != 0
        rho_p = rho_p[crit]
        out[crit] = 4 / (np.pi * delta ** 2) * ((1 + a6) * rho_p * jv(3, rho_p) - 6 * a6 * jv(4, rho_p)) / rho_p ** 4

        return out

    itg_x_y, x_cp, cp = numerical_kernel_integral(x_grid, y_grid, pdf)

    return itg_x_y, x_cp, cp


def gaussian_expansion(x_grid, y_grid, params):
    sigma = params[0]

    x_0, x_1, y_0, y_1 = get_scaled_variables(x_grid, y_grid, sigma)
    # x_0, x_1, y_0, y_1 = -1, 1, -1, 1
    x_y_itg_0 = 1 / 4 * (erf(x_1) - erf(x_0)) * (erf(y_1) - erf(y_0))
    # print(x_y_itg_0 - erf(1) ** 2)

    def e1(var):
        return var * np.exp(-var ** 2)

    x_y_itg_1 = 1 / (4 * np.sqrt(np.pi)) * ((erf(x_1) - erf(x_0)) * (e1(y_1) - e1(y_0)) +
                                            (erf(y_1) - erf(y_0)) * (e1(x_1) - e1(x_0)))
    # print(x_y_itg_1 - 2 * erf(1) / (np.e * np.sqrt(np.pi)))

    def e2(var):
        return var * (3 / 2 - var ** 2) * np.exp(-var ** 2)

    x_y_itg_2 = (1 / (8 * np.sqrt(np.pi)) * ((erf(x_1) - erf(x_0)) * (e2(y_1) - e2(y_0)) +
                                             (erf(y_1) - erf(y_0)) * (e2(x_1) - e2(x_0))) +
                 1 / (4 * np.pi) * (e1(x_1) - e1(x_0)) * (e1(y_1) - e1(y_0)))
    # print(x_y_itg_2 - (2 + np.e * np.sqrt(np.pi) * erf(1)) / (2 * np.e ** 2 * np.pi))

    # PDF profile
    x_cp, _ = upscale_samples(x_grid, 3)
    x_bar = x_cp / (np.sqrt(2) * sigma)

    alpha = params[1]
    if len(params) == 2:
        itg_x_y = x_y_itg_0 + alpha * x_y_itg_1
        cp = 1 / (2 * np.pi * sigma ** 2) * np.exp(- x_bar ** 2) * (1 + alpha * (1 - x_bar ** 2))

    elif len(params) == 3:
        beta = params[2]
        itg_x_y = x_y_itg_0 + alpha * x_y_itg_1 + beta * x_y_itg_2 / 2
        cp = 1 / (2 * np.pi * sigma ** 2) * np.exp(- x_bar ** 2) * (1 + alpha * (1 - x_bar ** 2) + beta / 2 * (1 - 2 * x_bar ** 2 + x_bar ** 4 / 2))

    else:
        sys.exit('Error: too many parameters')

    return itg_x_y, x_cp, cp


def get_scaled_variables(x_grid, y_grid, sigma):
    x_0 = x_grid[:-1, np.newaxis] / (np.sqrt(2) * sigma)
    x_1 = x_grid[1:, np.newaxis] / (np.sqrt(2) * sigma)
    y_0 = y_grid[np.newaxis, :-1] / (np.sqrt(2) * sigma)
    y_1 = y_grid[np.newaxis, 1:] / (np.sqrt(2) * sigma)
    return x_0, x_1, y_0, y_1


def numerical_kernel_integral(x_grid, y_grid, pdf_function, n_upscale=3):
    # Upscale the samples
    x_grid_up, idx_x_up = upscale_samples(x_grid, n_upscale)
    y_grid_up, idx_y_up = upscale_samples(y_grid, n_upscale)

    # Evaluate the PDF
    x_mesh, y_mesh = np.meshgrid(x_grid_up, y_grid_up, indexing='ij')
    pdf = pdf_function(x_mesh, y_mesh)
    # print(np.trapz(np.trapz(pdf, x=x_grid_up, axis=0), x=y_grid_up))

    # Integrate numerically
    itg_x_up = cumtrapz(pdf, x=x_grid_up, axis=0, initial=0)
    itg_x = itg_x_up[idx_x_up[1:], :] - itg_x_up[idx_x_up[:-1], :]

    itg_x_y_up = cumtrapz(itg_x, x=y_grid_up, axis=1, initial=0)
    itg_x_y = itg_x_y_up[:, idx_y_up[1:]] - itg_x_y_up[:, idx_y_up[:-1]]
    # print(np.sum(itg_x_y))

    idx_mid = np.argmin(np.abs(x_grid_up))
    pdf_mid = pdf[idx_mid, :]
    pdf_marginal = itg_x_up[-1, :]

    return itg_x_y, x_grid_up, pdf_mid


def upscale_samples(x, n):
    x_new = x.copy()
    for ii in range(n):
        x_mid = (x_new[1:] + x_new[:-1]) / 2
        idx_mid = np.arange(1, x_new.size)
        x_new = np.insert(x_new, idx_mid, x_mid)

    idx_initial = np.arange(0, x_new.size, 2 ** n)

    # Consistency check
    # print(x_new[idx_initial] - x)

    return x_new, idx_initial


def fwhm_bessel():
    delta = 1
    rho_max = 5
    n_rho = 1000
    rho = np.arange(1, n_rho + 1) / n_rho * rho_max
    a6 = np.linspace(-1.5, 2.5, 100)

    rho_mesh, a6_mesh = np.meshgrid(rho, a6, indexing='ij')

    f = 4 / (np.pi * delta ** 2) * ((1 + a6_mesh) * jv(3, rho_mesh / delta) / (rho_mesh / delta) ** 3
                                    - 6 * a6_mesh * jv(4, rho_mesh / delta) / (rho_mesh / delta) ** 4)

    # Consistency check: normalization
    # print(2 * np.pi * np.trapz(f * rho_mesh, x=rho, axis=0))

    half_max = (4 + a6) / (2 * 48 * np.pi * delta ** 2)

    f_flip = np.flip(f, axis=0)
    rho_flip = np.flip(rho)

    half_width = np.array([np.interp(half_max[ii], f_flip[:, ii], rho_flip) for ii in range(a6.size)])
    np.save('FWHM/a6_bessel.npy', a6)
    np.save('FWHM/fwhm_bessel.npy', 2 * half_width)

    half_width_quadratic_approx = (np.sqrt(10) * np.sqrt(4 + a6)) / np.sqrt(5 + 2 * a6)
    half_width_quartic_approx = 2 * np.sqrt(10 / (2 + a6) + 4 * a6 / (2 + a6) - np.sqrt(20 + 20 * a6 + 6 * a6 ** 2) / (2 + a6))

    fig, ax = plt.subplots()
    # ax.plot(rho, f)
    # ax.plot(half_width, half_max, 'x')
    ax.plot(a6, half_width)
    ax.plot(a6, half_width_quadratic_approx)
    # ax.plot(a6, half_width_quadratic_approx + np.mean(half_width - half_width_quadratic_approx))
    ax.plot(a6, half_width_quartic_approx)
    # ax.plot(a6, half_width_quartic_approx + np.mean(half_width - half_width_quartic_approx))
    plt.show()

    return 0


def fwhm_gaussian_expansion():
    sigma = 1
    rho_max = 5
    n_rho = 1001
    rho = np.arange(n_rho) / (n_rho - 1) * rho_max
    alpha = np.linspace(0, 1, 100)

    rho_mesh, alpha_mesh = np.meshgrid(rho, alpha, indexing='ij')

    f = (1 / (2 * np.pi * sigma ** 2) * np.exp(- rho_mesh ** 2 / (2 * sigma ** 2)) *
         (1 + alpha_mesh * (1 - rho_mesh ** 2 / (2 * sigma ** 2))))

    # Consistency check: normalization
    # print(2 * np.pi * np.trapz(f * rho_mesh, x=rho, axis=0))

    half_max = (1 + alpha) / (2 * 2 * np.pi * sigma ** 2)
    f_flip = np.flip(f, axis=0)
    rho_flip = np.flip(rho)

    half_width = np.array([np.interp(half_max[ii], f_flip[:, ii], rho_flip) for ii in range(alpha.size)])

    half_width_2 = np.zeros(alpha.shape)
    half_width_2[alpha < 0.001] = np.sqrt(2 * np.log(2) * (1 - alpha[alpha < 0.001]))
    alpha_g = alpha[alpha >= 0.001]
    half_width_2[alpha >= 0.001] = np.sqrt(2 + 2 / alpha_g - 2 * lambertw(np.exp(1 + 1 / alpha_g) * (1 + alpha_g) / (2 * alpha_g))) * sigma

    fig, ax = plt.subplots()
    # ax.plot(rho, f)
    # ax.plot(np.zeros(half_max.size), half_max, 'x')
    ax.plot(alpha, half_width)
    ax.plot(alpha, half_width_2)

    plt.show()


    return 0


if __name__ == '__main__':
    # fwhm_bessel()
    # fwhm_gaussian_expansion()
    pass
