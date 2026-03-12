"""
Contrast of the Derenzo phantom based on line profiles

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, x_valleys, y_valleys, radii, show_individual=False):

    h_0 = np.linspace(-2, 2, 101)[:, np.newaxis]
    v_0 = np.zeros(h_0.shape)
    h_60, v_60 = h_0 * 0.5, h_0 * np.sqrt(3) / 2
    h_120, v_120 = h_0 * (-0.5), h_0 * np.sqrt(3) / 2

    x_peaks_itp, y_peaks_itp = [], []
    x_valleys_itp, y_valleys_itp = [], []

    for ii in range(radii.size):
        x_peaks_itp.append(np.hstack((x_peaks[ii][np.newaxis, :] + h_0 * radii[ii],
                                      x_peaks[ii][np.newaxis, :] + h_60 * radii[ii],
                                      x_peaks[ii][np.newaxis, :] + h_120 * radii[ii])))

        y_peaks_itp.append(np.hstack((y_peaks[ii][np.newaxis, :] + v_0 * radii[ii],
                                      y_peaks[ii][np.newaxis, :] + v_60 * radii[ii],
                                      y_peaks[ii][np.newaxis, :] + v_120 * radii[ii])))

        x_valleys_itp_all = np.hstack((x_valleys[ii][np.newaxis, :] + h_0 * radii[ii],
                                       x_valleys[ii][np.newaxis, :] + h_60 * radii[ii],
                                       x_valleys[ii][np.newaxis, :] + h_120 * radii[ii]))

        y_valleys_itp_all = np.hstack((y_valleys[ii][np.newaxis, :] + v_0 * radii[ii],
                                       y_valleys[ii][np.newaxis, :] + v_60 * radii[ii],
                                       y_valleys[ii][np.newaxis, :] + v_120 * radii[ii]))

        # Only take the valley lines whose end points lie on the peak coordinates
        distance = np.sqrt((x_valleys_itp_all[0, :, np.newaxis] - x_peaks[ii][np.newaxis, :]) ** 2 +
                           (y_valleys_itp_all[0, :, np.newaxis] - y_peaks[ii][np.newaxis, :]) ** 2)
        valid = np.min(distance, axis=1) < 1e-12
        x_valleys_itp.append(x_valleys_itp_all[:, valid])
        y_valleys_itp.append(y_valleys_itp_all[:, valid])

    def derenzo_contrast_function(img):
        return derenzo_contrast(img, x_grid, y_grid, x_peaks_itp, y_peaks_itp, x_valleys_itp, y_valleys_itp, h_0.flatten(), radii, show_individual)

    def derenzo_contrast_slices_function(img):
        return derenzo_contrast_slices(derenzo_contrast_function, img)

    return derenzo_contrast_function, derenzo_contrast_slices_function


def derenzo_contrast(img, x_grid, y_grid, x_peaks_itp, y_peaks_itp, x_valleys_itp, y_valleys_itp, d, radii, show_invidivual):

    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2
    interpolator = RegularGridInterpolator((x, y), img)

    contrast_v1 = np.zeros(radii.shape)
    contrast_v2 = np.zeros(radii.shape)
    contrast_v3 = np.zeros(radii.shape)

    for ii in range(radii.size):
        peaks = interpolator((x_peaks_itp[ii], y_peaks_itp[ii]))
        valleys = interpolator((x_valleys_itp[ii], y_valleys_itp[ii]))

        peaks_mean = np.mean(peaks, axis=1)
        valleys_mean = np.mean(valleys, axis=1)

        contrast_v3[ii], fourier_cosine = continuous_fourier_cosine_transform(d * radii[ii], peaks_mean)
        # print(fourier_cosine)

        if show_invidivual:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
            im = ax0.imshow(img.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
            ax0.plot(x_peaks_itp[ii], y_peaks_itp[ii], color='tab:red')
            # ax0.plot(x_valleys_itp[ii], y_valleys_itp[ii], color='tab:red')
            cax = make_axes_locatable(ax0).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax0.set_xlabel(r'$x$ [mm]')
            ax0.set_ylabel(r'$y$ [mm]')

            ax1.plot(d * radii[ii], peaks, color='tab:blue', alpha=0.25)
            ax1.plot(d * radii[ii], peaks_mean, color='tab:blue', linewidth=3, label='Peaks')
            ax1.plot(d * radii[ii], valleys, color='tab:orange', alpha=0.25)
            ax1.plot(d * radii[ii], valleys_mean, color='tab:orange', linewidth=3, label='Valleys')
            # ax1.plot(d * radii[ii], fourier_cosine, color='tab:red', label='Fourier Cosine')

            ax1.set_xlim(-2 * radii[ii], 2 * radii[ii])
            ax1.set_aspect(4 * radii[ii] / np.diff(ax1.get_ylim()))
            ax1.set_xlabel(r'$d$ [mm]')
            ax1.legend()
            plt.show()

        i_p, i_v = np.interp(0., d * radii[ii], peaks_mean), np.interp(0., d * radii[ii], valleys_mean)
        # i_p, i_v = np.max(peaks_mean), np.min(valleys_mean)

        contrast_v1[ii] = (i_p - i_v) / (i_p + i_v)
        contrast_v2[ii] = (i_p - i_v) / i_p

    ground_truth_v3 = np.array([1.2571217665680090,
                                1.2486125009343059,
                                1.2289003578617081,
                                1.2063944760071232,
                                1.1949701041329437,
                                1.0821053640986047])

    contrast_v3 /= ground_truth_v3

    # fig, ax = plt.subplots()
    # ax.plot(1 / (4 * radii), contrast_v1)
    # # ax.plot(1 / (4 * radii), contrast_v2)
    # ax.set_ylim(-0.05, 1.05)
    # plt.show()

    return contrast_v3


def continuous_fourier_cosine_transform(x, y, n_max=1, show_expansion=False):
    ell = x[-1] - x[0]

    if len(y.shape) < 2:
        y = y[:, np.newaxis]

    a_cosine = np.zeros((y.shape[1], n_max + 1))
    cosine = np.zeros((n_max + 1, x.size))
    a_sine = np.zeros((y.shape[1], n_max + 1))
    sine = np.zeros((n_max + 1, x.size))

    for n in range(n_max + 1):
        a_cosine[:, n] = 2 / ell * np.trapz(y * np.cos(2 * np.pi * n * x[:, np.newaxis] / ell), x=x, axis=0)
        cosine[n, :] += np.cos(2 * np.pi * n * x / ell)
        a_sine[:, n] = 2 / ell * np.trapz(y * np.sin(2 * np.pi * n * x[:, np.newaxis] / ell), x=x, axis=0)
        sine[n, :] += np.sin(2 * np.pi * n * x / ell)
    a_cosine[:, 0] /= 2

    if show_expansion:
        plt.rcParams.update({'font.size': 24})
        fig, ax = plt.subplots()
        # ax.plot(x, y, alpha=0.5, label='Data')
        # ax.set_prop_cycle(None)
        # ax.plot(x, (a_cosine @ cosine).T, alpha=0.5, label='Individual expansions')

        lines = ax.plot(x, y, alpha=0.15, color='gray')
        lines[0].set_label('Profiles')
        # ax.plot(x, np.median(y, axis=1), color='tab:blue', linewidth=3, label='Median')
        ax.plot(x, (np.mean(a_cosine, axis=0)[np.newaxis, :] @ cosine).T, label='Mean FCS', color='tab:orange', linewidth=3)
        # ax.plot(x, (np.mean(a_sine, axis=0)[np.newaxis, :] @ sine).T, label='Mean FSS')
        ax.set_xlim(x[0], x[-1])
        # ax.set_ylim(0, 60)
        # ax.set_ylim(0, 60)
        ax.set_xlabel(r'$\ell$ [mm]')
        ax.set_ylabel('Image intensity')
        ax.legend()
        plt.show()

    contrast = a_cosine[:, n_max] / a_cosine[:, 0]
    # contrast = a_cosine[:, n_max] / a_cosine[:, 0] / (4 / np.pi)
    # contrast = a_sine[:, n_max] / a_cosine[:, 0]
    # contrast = (a_cosine[:, 1] + a_cosine[:, 2]) / a_cosine[:, 0]

    contrast_mean = np.mean(contrast)
    contrast_error = np.std(contrast)
    # contrast_error = np.percentile(contrast, q=75) - np.percentile(contrast, q=25)

    return contrast_mean, contrast_error


def derenzo_contrast_slices(derenzo_contrast_function, img):

    contrast = np.zeros((6, img.shape[-1]))
    for ii in range(img.shape[-1]):
        contrast[:, ii] = derenzo_contrast_function(img[:, :, ii])

    return contrast


if __name__ == "__main__":
    sys.exit()
