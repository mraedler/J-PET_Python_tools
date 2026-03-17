"""
Analysis of the modulation transfer function

Author: Martin Rädler
"""
# Python libraries
import sys
from pickle import dump, load
import numpy as np
from numpy.fft import fft, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import make_lsq_spline
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Auxiliary functions
from point_spread_functions_2d import get_2d_point_spread_function


def main():
    # x_grid = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/x_grid.npy')
    # y_grid = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/y_grid.npy')
    # img_peaks = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/img_peaks.npy')
    # # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/img_valleys.npy')
    # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/img_valleys_parzych.npy')
    # radii = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/radii.npy')

    # x_grid = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/x_grid.npy')
    # y_grid = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/y_grid.npy')
    # img_peaks = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/img_peaks.npy')
    # # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/img_valleys.npy')
    # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/img_valleys_parzych.npy')
    # radii = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_2.5/radii.npy')

    x_grid = np.load('/Derenzo_phantom/Derenzo_pixelated_3/x_grid.npy')
    y_grid = np.load('/Derenzo_phantom/Derenzo_pixelated_3/y_grid.npy')
    img_peaks = np.load('/Derenzo_phantom/Derenzo_pixelated_3/img_peaks.npy')
    # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated_3/img_valleys.npy')
    img_valleys = np.load('/Derenzo_phantom/Derenzo_pixelated_3/img_valleys_parzych.npy')
    radii = np.load('/Derenzo_phantom/Derenzo_pixelated_3/radii.npy')

    # plt.rcParams.update({'font.size': 12})
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    # ax0.imshow(np.sum(img_peaks, axis=2).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    # im = ax1.imshow(np.sum(img_valleys, axis=2).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    # cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.1)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # ax0.set_title('Peaks')
    # ax0.set_xlabel(r'$x$ [mm]')
    # ax0.set_ylabel(r'$y$ [mm]')
    # ax1.set_title('Valleys')
    # ax1.set_xlabel(r'$x$ [mm]')
    # plt.show()

    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2
    img_derenzo = np.sum(img_peaks, axis=-1)

    # #
    # pickle_file = open(sys.path[0] + '/Triangles_pixelated/mtf_correction.pkl', 'rb')
    # spline = load(pickle_file)
    # pickle_file.close()

    #
    point_spread_function, fwhm, _, _ = get_2d_point_spread_function('gaussian')

    #
    sigma_set_edges = np.linspace(0.35, 2.45, 22)
    sigma_set = (sigma_set_edges[1:] + sigma_set_edges[:-1]) / 2

    fwhm_set = fwhm(sigma_set)
    fwhm_rec = np.zeros(fwhm_set.shape)

    sf_fit = np.zeros((100, sigma_set.size))
    mtf_fit = np.zeros((100, sigma_set.size))

    sf_data = 1 / (4 * radii)
    mtf_data = np.zeros((6, sigma_set.size))

    cmap = plt.get_cmap("viridis", sigma_set.size)
    # norm = BoundaryNorm(np.arange(sigma_set.size + 1) + 0.5, sigma_set.size)
    norm = BoundaryNorm(sigma_set_edges, sigma_set.size)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for jj in range(sigma_set.size):
        # Blur the image
        psf, _, _ = point_spread_function(x, y, sigma_set[jj])
        img_conv = convolve2d(img_derenzo, psf, mode='same')

        # Modulation transfer function
        _, mtf_data[:, jj], sf_fit[:, jj], mtf_fit[:, jj] = get_derenzo_mtf(img_conv, img_peaks, img_valleys, radii)

        fwhm_rec[jj] = get_sigma(sf_fit[:, jj], mtf_fit[:, jj])

        ax.plot(sf_fit[:, jj], mtf_fit[:, jj], color=cmap(jj))
        ax.plot(sf_data, mtf_data[:, jj], 'x', color=cmap(jj))


    cbar = fig.colorbar(sm, ax=ax, ticks=sigma_set[::4], label=r'$\sigma$ [mm]')
    cbar.minorticks_off()

    idx_aa = np.argmin((1 - mtf_data[0, :]) ** 2 + mtf_data[-1, :] ** 2)
    ax.plot(sf_data, mtf_data[:, idx_aa], 'x', color='red')

    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Spatial frequency [1/mm]')
    ax.set_ylabel('Contrast')
    plt.show()

    print(fwhm_set[idx_aa])
    print(sigma_set[idx_aa])



    sys.exit()

    alpha = np.sum(fwhm_set * fwhm_rec) / np.sum(fwhm_rec ** 2)
    beta = np.mean(fwhm_set - fwhm_rec)
    # print(beta)

    fig, ax = plt.subplots()
    ax.plot(fwhm_set)
    ax.plot(fwhm_rec)
    # ax.plot(fwhm_rec * alpha)
    # ax.plot(fwhm_rec + beta)
    ax.set_ylim(bottom=0)
    plt.show()


    return 0


def get_derenzo_mtf(img, img_peaks, img_valleys, radii):
    # Evaluate the contrast
    sf_data = 1 / (4 * radii)
    mtf_data = evaluate_contrast_derenzo(img, img_peaks, img_valleys)

    # Correct negative contrast due to boundary effects
    mtf_data += fitted_function(mtf_data)

    # # Correction due pixel size
    # plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.plot(1 / (4 * radii), evaluate_contrast_derenzo(np.sum(img_peaks, axis=-1), img_peaks, img_valleys, kind='median'), label='Weighted median')
    # ax.plot(1 / (4 * radii), evaluate_contrast_derenzo(np.sum(img_peaks, axis=-1), img_peaks, img_valleys, kind='mean'), label='Weighted mean')
    # ax.set_ylim(-0.05, 1.05)
    # ax.set_xlabel('Spatial frequency [1/mm]')
    # ax.set_ylabel('Contrast')
    # ax.legend(loc='lower center')
    # plt.show()

    # mtf_data *= 1 / evaluate_contrast_derenzo(np.sum(img_peaks, axis=-1), img_peaks, img_valleys)
    # mtf_data *= np.array([1., 1., 1., 1., 1., 1.03644094])

    # Fit
    sf_fit, mtf_fit = constrained_polynomial_fit(sf_data, mtf_data, deg=5, vis=False)
    # sf_fit, mtf_fit = center_gaussian_fit(sf_data, mtf_data, vis=False)
    # sf_fit, mtf_fit = polynomial_plus_tail_fit(sf_data, mtf_data, deg=3)
    return sf_data, mtf_data, sf_fit, mtf_fit


def main2():
    #
    x_grid = np.load(sys.path[0] + '/Triangles_pixelated/x_grid.npy')
    y_grid = np.load(sys.path[0] + '/Triangles_pixelated/y_grid.npy')
    x = np.load(sys.path[0] + '/Triangles_pixelated/x.npy')
    y = np.load(sys.path[0] + '/Triangles_pixelated/y.npy')

    #
    distances = np.array([10.0, 8.0, 6.0, 5.0, 4.0, 3.2])  # mm
    n_rows_original = np.array([3, 4, 5, 6, 7, 8])
    n_rows_padded = np.array([9, 12, 16, 20, 25, 31])

    #
    point_spread_function, fwhm, _, _ = get_2d_point_spread_function('gaussian')
    # point_spread_function, fwhm, _, _ = get_2d_point_spread_function('lorentzian')
    # point_spread_function, fwhm, _, _ = get_2d_point_spread_function('polynomial')
    # fwhm_set = fwhm([sigma_set])
    sigma_set = np.linspace(0.5, 2.5, 17)

    contrast_original, contrast_padded = [], []
    for jj in trange(sigma_set.size):

        psf, _, _ = point_spread_function(x, y, sigma_set[jj])
        # psf, _, _ = point_spread_function(x, y, [3 * sigma_set[jj], 0])

        # fig, ax = plt.subplots()
        # ax.imshow(psf.T, origin='lower')
        # plt.show()

        aa_temp = np.zeros(distances.shape)
        bb_temp = np.zeros(distances.shape)

        for ii in range(distances.size):
            img_peaks_original = np.load(sys.path[0] + '/Triangles_pixelated/img_peaks_%1.1fmm_%drows.npy' % (distances[ii], n_rows_original[ii]))
            img_peaks_padded = np.load(sys.path[0] + '/Triangles_pixelated/img_peaks_%1.1fmm_%drows.npy' % (distances[ii], n_rows_padded[ii]))
            img_valleys_original = np.load(sys.path[0] + '/Triangles_pixelated/img_valleys_%1.1fmm_%drows.npy' % (distances[ii], n_rows_original[ii]))

            img_conv_original = convolve2d(img_peaks_original, psf, mode='same')
            img_conv_padded = convolve2d(img_peaks_padded, psf, mode='same')

            aa_temp[ii] = evaluate_contrast(img_conv_original, img_peaks_original, img_valleys_original)
            bb_temp[ii] = evaluate_contrast(img_conv_padded, img_peaks_original, img_valleys_original)

            # plt.rcParams.update({'font.size': 12})
            # fig, (ax0, ax1) = plt.subplots(1, 2)
            # ax0.imshow(img_peaks_original.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
            # ax0.set_title('Original')
            # ax1.imshow(img_peaks_padded.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
            # ax1.set_title('Expanded')
            # ax0.set_xlabel(r'$x$ [mm]')
            # ax0.set_ylabel(r'$y$ [mm]')
            # ax1.set_xlabel(r'$x$ [mm]')
            # plt.show()

        contrast_original.append(aa_temp)
        contrast_padded.append(bb_temp)

    contrast_original = np.concatenate(contrast_original)
    contrast_padded = np.concatenate(contrast_padded)
    idx = np.argsort(contrast_original)
    contrast_original = contrast_original[idx]
    contrast_padded = contrast_padded[idx]

    correction = contrast_padded - contrast_original

    # a = 0.042
    a = 0.071
    # a = 0.155

    # correction_extended
    delta = 0.01
    left_extension = np.arange(-50, 0) * delta + contrast_original[0]
    right_extension = np.arange(1, 50 + 1) * delta + contrast_original[-1]

    # correction = correction[contrast_original < 0.7]
    # contrast_original = contrast_original[contrast_original < 0.7]

    contrast_original_extended = np.concatenate((left_extension, contrast_original, right_extension))
    correction_extended = np.concatenate((-left_extension, correction, np.zeros(right_extension.shape)))

    # # Fit B-spline using least squares
    k = 3
    knots = np.concatenate((np.ones(k + 1) * contrast_original_extended[0],
                            [-2 * a, -a, -a, -a, -a, 0., 0.1, 0.2, 0.4, 0.5, 0.6, 1., 1., 1., 1.],
                            np.ones(k + 1) * contrast_original_extended[-1]))

    # spline = make_lsq_spline(contrast_original_extended, correction_extended, knots, k=k)
    # pickle_file = open(sys.path[0] + '/Triangles_pixelated/mtf_correction.pkl', 'wb')
    # dump(spline, pickle_file)
    # pickle_file.close()

    p_opt, p_cov = curve_fit(lambda var, x0: fit_function(var, x0, a), contrast_original, correction, p0=(-2 * a))
    print(p_opt)

    # pickle_file = open(sys.path[0] + '/Triangles_pixelated/mtf_correction_median.pkl', 'wb')
    # dump(fitted_function, pickle_file)
    # pickle_file.close()

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    ax.plot(contrast_original, correction, 'x', alpha=1, label='Data')
    # ax.plot(contrast_original_extended, correction_extended, 'o', fillstyle='none')
    ax.plot(contrast_original_extended, -contrast_original_extended, '--', label=r'$-C$', color='tab:green')

    x = np.linspace(-a, 1.1, 100)
    ax.plot(x, fitted_function(x), '-', label=r'$-C_0\exp\left(\dfrac{C-C_0}{C_0}\right)$', color='tab:orange')


    # ax.plot(knots, np.zeros(knots.shape), 'o')
    # ax.plot(x, spline(x))

    ax.set_xlim(-0.2, 1.1)
    ax.set_ylim(-0.01, 0.08)
    # ax.set_aspect(1)

    ax.set_xlabel(r'Original contrast $C$')
    ax.set_ylabel('Difference to the contrast from the expanded image')
    ax.legend(loc='upper right')

    plt.show()

    return 0


def fit_function(var, x0, a):
    return a * ((var - x0) / (-a - x0)) ** ((a + x0) / a)


def fitted_function(var):
    a = 0.071
    p_opt = -817.13931351
    # p_opt = -5062.49990223
    # a = 0.155
    # p_opt = -8061.497593
    out = np.zeros(var.shape)
    out[var <= -a] = - var[var <= -a]
    # out[(var > -a) & (var < 1.)] = fit_function(var[(var > -a) & (var < 1.)], p_opt, a)
    out[(var > -a) & (var < 1.)] = a * np.exp(-var[(var > -a) & (var < 1.)] / a - 1)
    out[var >= 1.] = 0.
    return out


def evaluate_contrast(img, img_peaks, img_valleys):
    # # Mean based
    # peaks = np.sum(img * img_peaks) / np.sum(img_peaks)
    # valleys = np.sum(img * img_valleys) / np.sum(img_valleys)

    # Median based
    peaks = weighted_median(img, img_peaks)
    valleys = weighted_median(img, img_valleys)

    return (peaks - valleys) / (peaks + valleys)


def evaluate_contrast_derenzo(img, img_peaks, img_valleys, kind='median'):

    if kind == 'mean':
        peaks = np.sum(img[:, :, np.newaxis] * img_peaks, axis=(0, 1)) / np.sum(img_peaks, axis=(0, 1))
        valleys = np.sum(img[:, :, np.newaxis] * img_valleys, axis=(0, 1)) / np.sum(img_valleys, axis=(0, 1))

    elif kind == 'median':
        peaks = np.zeros(img_peaks.shape[-1])
        valleys = np.zeros(img_valleys.shape[-1])

        for ii in range(img_peaks.shape[-1]):
            peaks[ii] = weighted_median(img, img_peaks[:, :, ii])
            valleys[ii] = weighted_median(img, img_valleys[:, :, ii])

    else:
        sys.exit('Unknown kind.')

    return (peaks - valleys) / (peaks + valleys)


def weighted_median(img, weights):

    x = img.flatten()
    idx_sort = np.argsort(x)
    x = x[idx_sort]

    weights = weights.flatten() / np.sum(weights)
    weights = weights[idx_sort]

    idx = np.round(np.interp(0.5, np.cumsum(weights), np.arange(weights.size))).astype(int)

    return x[idx]


def constrained_polynomial_fit(x_data, y_data, deg=5, vis=False):

    def polynomial(x, *params):
        a4_onwards = np.array(params[:-1])
        delta = params[-1]
        integers = np.arange(a4_onwards.size)
        a0_to_a3 = [1, 0, -3 + np.sum((integers + 1) * a4_onwards), 2 - np.sum((integers + 2) * a4_onwards)]
        ai = np.concatenate((a0_to_a3, a4_onwards))[:, np.newaxis]

        power = np.arange(ai.size)[:, np.newaxis]

        return np.sum(ai * (x / delta) ** power, axis=0) * np.heaviside(1 - x / delta, 0.5)

    # Using curve fit
    p_0 = np.append(np.zeros(deg - 3), x_data[-1])
    infs = np.full(deg - 3, np.inf)
    p_opt, p_cov = curve_fit(polynomial, x_data, y_data, p0=p_0, bounds=([*(-infs), 0], [*infs, 10 * x_data[-1]]), maxfev=1000)
    # print(p_opt)

    x_samples = np.linspace(0, 3 * x_data[-1], 100)
    y_samples_fit = polynomial(x_samples, *p_opt)

    # Using minimize with Nelder-Mead algorithm
    parameter_steps = np.append(10 * np.ones(deg - 3), x_data[-1])
    initial_simplex = np.repeat(p_0[np.newaxis, :], p_0.size + 1, axis=0)
    initial_simplex[1:, :] += np.diag(parameter_steps)

    def objective_function(params):
        # Exclude results with values below zero or above one
        y_samples = polynomial(x_samples, *params)
        if np.any(y_samples > 1) | np.any(y_samples < 0):
            return np.nan

        # Exclude results with more than one turning points
        second_derivative = y_samples[2:] - 2 * y_samples[1:-1] + y_samples[:-2]
        if np.count_nonzero(second_derivative[1:] * second_derivative[:-1] < 0.) > 1:
            return np.nan

        return np.sum((polynomial(x_data, *params) - y_data) ** 2)

    res = minimize(objective_function, p_0, method='Nelder-Mead', options={'initial_simplex': initial_simplex})
    # print(res.x)

    y_samples_min = polynomial(x_samples, *res.x)

    if vis:
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, 'x')
        ax.plot(x_samples, y_samples_fit)
        ax.plot(x_samples, y_samples_min, '--')
        ax.set_xlim(0, x_data[-1])
        ax.set_ylim(-0.1, 1.1)
        plt.show()

    # return x_samples, y_samples_fit
    return x_samples, y_samples_min


def center_gaussian_fit(x_data, y_data, vis=False):

    def center_gaussian(x, sigma):
        # return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))
        return np.exp(-x ** 2 / (2 * sigma ** 2))

    p_opt, p_cov = curve_fit(center_gaussian, x_data, y_data, p0=.1)

    x_samples = np.linspace(0, 7 * p_opt[0], 100)
    y_samples = center_gaussian(x_samples, *p_opt)

    return x_samples, y_samples


def polynomial_plus_tail_fit(x_data, y_data, deg=2):
    #

    def polynomial_plus_tail(x, *params):
        # Separate the parameters
        ai = np.append([1, 0], params[:-2])
        delta = params[-2]
        n = params[-1]

        # Allocate
        y = np.zeros(x.shape)

        # Polynomial
        power = np.arange(ai.size)
        y[x < delta] = np.sum(ai * (x[x < delta, np.newaxis] / delta) ** power, axis=1)

        # Power law / exponential tail
        y_0 = np.sum(ai)
        y_1 = np.sum(np.arange(1, ai.size) * ai[1:])
        y[x >= delta] = y_0 * (1 - y_1 / y_0 * (x[x >= delta] / delta - 1) / n) ** (-n)

        return y

    # Fit
    p_0 = np.append(np.zeros(deg - 1), [x_data[-1] / 2, 3])
    p_0[0] = -1 / 2
    bounds = ([*np.full(deg - 1, -np.inf), 0, 1], [*np.full(deg - 1, np.inf), np.inf, np.inf])
    p_opt, p_cov = curve_fit(polynomial_plus_tail, x_data, y_data, p0=p_0, bounds=bounds)

    # Todo: add Nelder-Mead approach
    print(p_opt)


    x_samples = np.linspace(0, 1.1 * x_data[-1], 100)
    y_samples = polynomial_plus_tail(x_samples, *p_opt)

    return x_samples, y_samples


def get_sigma(x, y):
    x_sym = np.concatenate((-np.flip(x), x[1:]))
    y_sym = np.concatenate((np.flip(y), y[1:]))
    p_sym = y_sym / np.trapz(y_sym, x=x_sym)

    x_sym = x
    p_sym = y / np.trapz(y, x=x_sym)

    mtf_pdf_estimate_sigma = 1 / (2 * np.pi * np.sqrt(np.trapz(p_sym * (x_sym ** 2), x=x_sym)))
    mtf_pdf_estimate_fwhm = 2 * np.sqrt(2 * np.log(2)) * mtf_pdf_estimate_sigma
    # mtf_def_estimate_fwhm = 2 * np.sqrt(2 * np.log(2)) / (2 * np.pi * np.interp(0.5, np.flip(y), np.flip(x)) / (np.sqrt(2 * np.log(2))))
    mtf_def_estimate_fwhm = 4 * np.log(2) / (2 * np.pi * np.interp(0.5, np.flip(y), np.flip(x)))

    p_sym = np.pad(p_sym, (1000, 1000), 'constant', constant_values=(0, 0))
    p_sym_fft = fftshift(fft(ifftshift(p_sym)))

    average_imag = np.mean(np.abs(np.imag(p_sym_fft)))
    if average_imag > 1e-12:
        print('Warning: imaginary part above threshold!')
    p_sym_fft = p_sym_fft.real

    f_fft = (np.arange(p_sym_fft.size) / p_sym_fft.size) / (x[1] - x[0])
    idx = np.round((p_sym_fft.size - 1) / 2).astype(int)
    f_fft -= f_fft[idx]

    # p_sym_fft /= np.trapz(p_sym_fft, x=f_fft)

    fig, ax = plt.subplots()
    ax.plot(f_fft, p_sym_fft / np.max(p_sym_fft))
    plt.show()

    psf_pdf_estimate_sigma = np.sqrt(np.trapz(p_sym_fft * (f_fft ** 2), x=f_fft))
    psf_pdf_estimate_fwhm = 2 * np.sqrt(2 * np.log(2)) * psf_pdf_estimate_sigma
    psf_def_estimate_fwhm = 2 * np.interp(0.5, np.flip(p_sym_fft[idx:] / p_sym_fft[idx]), np.flip(f_fft[idx:]))

    # print(mtf_pdf_estimate_sigma)
    # print(mtf_pdf_estimate_fwhm)
    # print(mtf_def_estimate_fwhm)

    # print(psf_pdf_estimate_sigma)
    # print(psf_pdf_estimate_fwhm)
    # print(psf_def_estimate_fwhm)

    # print()

    return (mtf_def_estimate_fwhm + psf_def_estimate_fwhm) / 2
    # return psf_def_estimate_fwhm
    # return mtf_def_estimate_fwhm
    # return mtf_def_estimate_fwhm


if __name__ == "__main__":
    main()
    # main2()
