"""
Contrast of the Derenzo phantom based on line profiles

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import MultiPoint, Point
from shapely.affinity import scale
from shapely.prepared import prep

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=False):
    h_0 = np.linspace(-2, 2, 101)[:, np.newaxis]
    v_0 = np.zeros(h_0.shape)
    h_60, v_60 = h_0 * 0.5, h_0 * np.sqrt(3) / 2
    h_120, v_120 = h_0 * (-0.5), h_0 * np.sqrt(3) / 2

    x_peaks_itp, y_peaks_itp, hulls = [], [], []

    for ii in range(radii.size):
        #
        hull = MultiPoint(np.stack((x_peaks[ii], y_peaks[ii]), axis=1)).convex_hull
        hull_scaled = scale(hull, 1.05, 1.05, origin='centroid')
        hulls.append(hull_scaled)
        hull_prepared = prep(hull_scaled)

        x_peaks_itp_temp = []
        y_peaks_itp_temp = []

        invert_selection = False

        for jj in range(x_peaks[ii].size):
            points_x_0, points_y_0 = x_peaks[ii][jj] + h_0 * radii[ii], y_peaks[ii][jj] + v_0 * radii[ii]
            points_validity = [hull_prepared.contains(Point(x_p, y_p)) for x_p, y_p in zip(points_x_0, points_y_0)]
            if (all(points_validity) or use_all_profiles) ^ invert_selection:
                x_peaks_itp_temp.append(points_x_0)
                y_peaks_itp_temp.append(points_y_0)

            points_x_60, points_y_60 = x_peaks[ii][jj] + h_60 * radii[ii], y_peaks[ii][jj] + v_60 * radii[ii]
            points_validity = [hull_prepared.contains(Point(x_p, y_p)) for x_p, y_p in zip(points_x_60, points_y_60)]
            if (all(points_validity) or use_all_profiles) ^ invert_selection:
                x_peaks_itp_temp.append(points_x_60)
                y_peaks_itp_temp.append(points_y_60)

            points_x_120, points_y_120 = x_peaks[ii][jj] + h_120 * radii[ii], y_peaks[ii][jj] + v_120 * radii[ii]
            points_validity = [hull_prepared.contains(Point(x_p, y_p)) for x_p, y_p in zip(points_x_120, points_y_120)]
            if (all(points_validity) or use_all_profiles) ^ invert_selection:
                x_peaks_itp_temp.append(points_x_120)
                y_peaks_itp_temp.append(points_y_120)

        x_peaks_itp.append(np.hstack(x_peaks_itp_temp))
        y_peaks_itp.append(np.hstack(y_peaks_itp_temp))

        # fig, ax = plt.subplots()
        # ax.plot(x_peaks_itp[-1], y_peaks_itp[-1])
        # ax.plot(*hull_scaled.exterior.xy, linestyle='--', linewidth=3, color='k')
        # plt.show()

    # correction_wave_numbers = np.load(sys.path[0] + '/Pixelated_ground_truth/correction_wave_numbers.npy')
    # correction_factors = np.load(sys.path[0] + '/Pixelated_ground_truth/correction_factors.npy')

    path = "/home/martin/PycharmProjects/J-PET_Python_tools/Derenzo_phantom"
    correction_wave_numbers = np.load(path + '/Pixelated_ground_truth/correction_wave_numbers.npy')
    correction_factors = np.load(path + '/Pixelated_ground_truth/correction_factors.npy')

    def derenzo_contrast_function_2d(img_2d):
        # show_interpolation_lines(x_grid, y_grid, img_2d, x_peaks_itp, y_peaks_itp)
        return derenzo_contrast(img_2d, x_grid, y_grid, x_peaks_itp, y_peaks_itp, radii, h_0.flatten(), correction_wave_numbers, correction_factors, apply_correction=apply_correction, highest_order=highest_order)

    def derenzo_contrast_function_3d(img_3d):

        wave_numbers_collected, contrast_values_collected, contrast_errors_collected = [], [], []

        for ii in range(img_3d.shape[-1]):
            wave_numbers, contrast_values, contrast_errors = (
                derenzo_contrast(img_3d[:, :, ii], x_grid, y_grid, x_peaks_itp, y_peaks_itp, radii, h_0.flatten(), correction_wave_numbers, correction_factors, apply_correction=apply_correction, highest_order=highest_order))

            wave_numbers_collected.append(wave_numbers)
            contrast_values_collected.append(contrast_values)
            contrast_errors_collected.append(contrast_errors)

        wave_numbers_collected = np.vstack(wave_numbers_collected)
        contrast_values_collected = np.vstack(contrast_values_collected)
        contrast_errors_collected = np.vstack(contrast_errors_collected)

        wave_numbers_collected = np.mean(wave_numbers_collected, axis=0)
        contrast_values_collected = np.mean(contrast_values_collected, axis=0)
        contrast_errors_collected = np.sqrt(np.sum(contrast_errors_collected ** 2, axis=0)) / img_3d.shape[-1]

        return wave_numbers_collected, contrast_values_collected, contrast_errors_collected

    return derenzo_contrast_function_2d, derenzo_contrast_function_3d


def derenzo_contrast(img, x_grid, y_grid, x_peaks_itp, y_peaks_itp, radii, distance, correction_wave_numbers, correction_factors, apply_correction=False, highest_order=20):
    # Linear interpolator
    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2
    interpolator = RegularGridInterpolator((x, y), img)

    # Fourier series coefficients of an ideal box
    # It does not make sense to compare the zero frequency due to the normalization
    # (only odd entries are non-zero in the expansion of a box function, therefore step by two)
    nn = np.arange(1, highest_order + 1, 2)[:, np.newaxis]
    ideal_contrast = 2 / (np.pi * nn) * np.sin(np.pi * nn / 2)

    wave_numbers = []
    contrast_values = []
    contrast_errors = []

    for ii in range(radii.size):
        # Get the line profiles by interpolation
        peaks = interpolator((x_peaks_itp[ii], y_peaks_itp[ii]))
        ax1 = False
        # ax1 = show_derenzo_interpolation(x_grid, y_grid, img, x_peaks_itp[ii], y_peaks_itp[ii], radii[ii], distance, peaks)

        # Fourier series
        k_n, cosine_coefficients, sine_coefficients = fourier_series(radii[ii] * distance, peaks, highest_order, ax=ax1)

        # Measure of how asynchronous it is
        # print(np.mean(np.abs(sine_coefficients / (cosine_coefficients[0:1, :] * 2))[1, :]))

        # Coefficient strength relative to the zero frequency (to normalize)
        # cosine_coefficients[0:1, :] is multiplied by 2 because in fourier_series it is divided by 2
        # Only include every second (nonzero) frequency
        contrast = cosine_coefficients[1::2, :] / (cosine_coefficients[0:1, :] * 2)
        normalized_contrast = contrast / ideal_contrast

        # nn_plot = np.linspace(nn[0][0], nn[-1][0], 100)
        # plt.rcParams.update({'font.size': 24})
        # fig, ax = plt.subplots()
        # ax.plot(nn_plot, 2 / (np.pi * nn_plot) * np.sin(np.pi * nn_plot / 2), color='tab:blue', alpha=0.2)
        # ax.stem(nn, np.mean(ideal_contrast, axis=1), linefmt='tab:blue', basefmt='k', label=r'$\dfrac{2}{\pi n}\sin\left(\dfrac{\pi n}{2}\right)$')
        # ax.stem(nn, np.mean(contrast, axis=1), linefmt='tab:orange', basefmt='k', label=r'$\dfrac{\tilde{g}_n}{\tilde{g}_0}$')
        # ax.set_ylim(-0.27, 0.67)
        # ax.set_xticks(nn.flatten())
        # ax.set_xlabel(r'$n$')
        # ax.legend(frameon=False)
        # plt.show()

        wave_numbers.append(k_n[1::2])
        contrast_values.append(np.mean(normalized_contrast, axis=1))
        contrast_errors.append(np.std(normalized_contrast, axis=1))

    # Concatenate list to numpy array
    wave_numbers = np.concatenate(wave_numbers).flatten()
    contrast_values = np.concatenate(contrast_values)
    contrast_errors = np.concatenate(contrast_errors)

    # Sort with respect to the wave number
    idx_sort = np.argsort(wave_numbers)
    wave_numbers = wave_numbers[idx_sort]
    contrast_values = contrast_values[idx_sort]
    contrast_errors = contrast_errors[idx_sort]

    if apply_correction:
        contrast_values /= np.interp(wave_numbers, correction_wave_numbers, correction_factors)

    return wave_numbers, contrast_values, contrast_errors


def show_derenzo_interpolation(x_grid, y_grid, img, x_peaks_itp, y_peaks_itp, radius, d, peaks):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    im = ax0.imshow(img.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax0.plot(x_peaks_itp, y_peaks_itp, color='tab:orange')
    ax0.plot([x_peaks_itp[0], x_peaks_itp[-1]], [y_peaks_itp[0], y_peaks_itp[-1]], marker='.', markersize=4, color='tab:orange')
    cax = make_axes_locatable(ax0).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')

    return ax1


def show_interpolation_lines(x_grid, y_grid, img, x_peaks_itp, y_peaks_itp):
    # plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'font.size': 24})
    fig, ax= plt.subplots(figsize=(6, 5))
    im = ax.imshow(img.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ii in range(len(x_peaks_itp)):
        # ax.plot(x_peaks_itp[ii], y_peaks_itp[ii], color='tab:red')
        # ax.plot(x_peaks_itp[ii], y_peaks_itp[ii], color=colors[ii])
        ax.plot([x_peaks_itp[ii][0], x_peaks_itp[ii][-1]], [y_peaks_itp[ii][0], y_peaks_itp[ii][-1]], marker='.', markersize=4, color=colors[ii])

    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)

    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')

    # ax1.plot(d * radius, peaks, color='tab:red', alpha=0.25)
    # ax1.set_xlim(-2 * radius, 2 * radius)
    # ax1.set_aspect(4 * radius / np.diff(ax1.get_ylim()))
    # ax1.set_xlabel(r'$d$ [mm]')
    # # ax1.legend()
    plt.show()

    return 0


def fourier_series(x, y, highest_order, ax=False):
    ell = x[-1] - x[0]

    if len(y.shape) < 2:
        y = y[:, np.newaxis]

    enn = np.arange(highest_order + 1)
    cosine_coefficients = np.zeros((enn.size, y.shape[1]))
    cosine_bases = np.zeros((x.size, enn.size))
    sine_coefficients = np.zeros((enn.size, y.shape[1]))
    sine_bases = np.zeros((x.size, enn.size))

    for n in enn:  #
        cosine_coefficients[n, :] = 2 / ell * np.trapezoid(y * np.cos(2 * np.pi * n * x[:, np.newaxis] / ell), x=x, axis=0)
        cosine_bases[:, n] += np.cos(2 * np.pi * n * x / ell)
        sine_coefficients[n, :] = 2 / ell * np.trapezoid(y * np.sin(2 * np.pi * n * x[:, np.newaxis] / ell), x=x, axis=0)
        sine_bases[:, n] += np.sin(2 * np.pi * n * x / ell)
    cosine_coefficients[0, :] /= 2

    if ax:
        cosine_expansion = cosine_bases @ cosine_coefficients
        sine_expansion = sine_bases @ sine_coefficients

        # fig, ax = plt.subplots()
        lines = ax.plot(x, y, alpha=0.5, color='orange')
        # ax.plot(x, np.mean(y, axis=1), color='tab:blue', linewidth=2, label='Mean')
        # ax.plot(x, cosine_expansion)
        # ax.plot(x, sine_expansion)
        ax.plot(x, np.mean(cosine_expansion, axis=1), label='Mean FCS', linestyle='-', color='tab:blue', linewidth=2)
        # ax.plot(x, np.mean(sine_expansion, axis=1), label='Mean FSS', linestyle='--', color='tab:red', linewidth=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        # ax.set_xlabel(r'$\ell$ [mm]')
        ax.set_xlabel(r'$s$ [mm]')
        ax.set_ylabel('Image intensity')
        lines[0].set_label('Profiles')
        ax.legend(loc='lower center', frameon=False)
        plt.show()

    k_enn = 2 * np.pi * enn / ell

    return k_enn, cosine_coefficients, sine_coefficients


if __name__ == "__main__":
    sys.exit()
