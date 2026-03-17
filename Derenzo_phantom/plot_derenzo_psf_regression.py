"""
Plot the convergence of the Point Spread Function (PSF) regression

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize._optimize import OptimizeResult
import matplotlib.pyplot as plt

# Auxiliary functions
from Derenzo_phantom.get_derenzo_image import get_ground_truth_derenzo_image
from Derenzo_phantom.get_derenzo_contrast import get_derenzo_contrast_function
from Derenzo_phantom.get_derenzo_image import load_derenzo_image
from Derenzo_phantom.psf_mtf_regression import fit_psf_2d, blur_ground_truth


def main():
    # Load
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    derenzo_contrast_2d, derenzo_contrast_3d = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, radii, highest_order=20, use_all_profiles=False, apply_correction=True)

    img_derenzo = np.sum(img_peaks, axis=-1)
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_true')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_energy')

    idx_it_200, = np.where(iterations == 200)[0]
    img_recon = np.mean(imgs[:, :, :, idx_it_200], axis=-1)
    distances, profiles, x_peaks_itp, y_peaks_itp = derenzo_contrast_2d(img_recon, return_profiles=True)

    # fig, ax = plt.subplots()
    # ax.imshow(img_recon.T, origin='lower')
    # # ax.imshow(img_peaks.T, origin='lower')
    # plt.show()

    model = 'hermite-gaussian'
    # model = 'plateau-polynomial'

    p_opt = fit_psf_2d(x, y, img_recon, img_derenzo, model=model, show_convergence=False)
    # p_opt = OptimizeResult()
    # p_opt.x = np.array([1.17877477, 1., 194.51014175])
    print(p_opt.x)

    scaled_blurred_ground_truth_psf_fit = p_opt.x[-1] * blur_ground_truth(x, y, img_derenzo, model, p_opt.x[:-1], n_cut=150)

    blurred_ground_truth_mtf_fit = blur_ground_truth(x, y, img_derenzo, model, [1.60176445, 1.])
    # blurred_ground_truth_mtf_fit = blur_ground_truth(x, y, img_derenzo, model, [1.14565584, 0.65155488], n_cut=150)
    amplitude_mtf_fit = np.sum(img_recon) / np.sum(blurred_ground_truth_mtf_fit)
    # amplitude_mtf_fit = np.sum(blurred_ground_truth_mtf_fit * img_recon) / np.sum(blurred_ground_truth_mtf_fit ** 2)
    scaled_blurred_ground_truth_mtf_fit = amplitude_mtf_fit * blurred_ground_truth_mtf_fit

    scaled_ground_truth = np.sum(img_recon) / np.sum(img_derenzo) * img_derenzo

    # print(np.mean(scaled_blurred_ground_truth_psf_fit))
    n_cylinders = np.array([xp.size for xp in x_peaks])
    print(np.sum(np.pi * radii ** 2 * n_cylinders / (0.5 * 0.5)))
    print(np.sum(img_derenzo))
    print(np.sum(img_recon))
    print(np.sum(img_recon) / np.sum(img_derenzo))
    # print(np.mean(blur_ground_truth(x, y, img_derenzo, model, p_opt.x[:-1], n_cut=150)))
    # sys.exit()

    # print(p_opt.x[-1])
    # print(amplitude_mtf_fit)

    images_list = [img_recon, scaled_ground_truth, scaled_blurred_ground_truth_psf_fit, scaled_blurred_ground_truth_mtf_fit]

    # compare_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii, idx=2)
    # compare_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii, idx=5)

    # compare_four_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii)
    compare_four_derenzo_images_and_profiles_v2(x_grid, y_grid, images_list, radii, derenzo_contrast_2d)

    return 0


def compare_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii, idx=0):
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    triangle_numbers = [int((-1 + np.sqrt(1 + 8 * len(x_p))) / 2) for x_p in x_peaks]

    # x_0, y_0 = x_peaks[idx][-1], y_peaks[idx][-1]
    # x_0, y_0 = x_peaks[idx][0], y_peaks[idx][0]
    x_0, y_0 = x_peaks[idx][1], y_peaks[idx][1]
    # x_1, y_1 = x_peaks[idx][-triangle_numbers[idx]], y_peaks[idx][-triangle_numbers[idx]]
    x_1, y_1 = x_peaks[idx][-2], y_peaks[idx][-2]

    alpha = np.linspace(-0.2, 1.2, 100)
    x_itp = x_0 * (1 - alpha) + x_1 * alpha
    y_itp = y_0 * (1 - alpha) + y_1 * alpha

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_lim = [np.min(images_list[0]), np.max(images_list[0])]
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, len(images_list) + 1, figsize=(12, 4))
    for ii in range(len(images_list)):

        interpolator = RegularGridInterpolator((x, y), images_list[ii])
        itp_img = interpolator((x_itp, y_itp))

        axes[ii].imshow(images_list[ii].T, origin='lower', extent=extent, clim=c_lim)
        # axes[ii].plot(x_0, y_0, 'x', color=colors[ii])
        # axes[ii].plot(x_1, y_1, 'x', color=colors[ii])
        axes[ii].plot(x_itp, y_itp, color=colors[ii])

        # axes[ii].set_xticks([-50, 0, 50])
        # axes[ii].set_yticks([-50, 0, 50])

        axes[ii].set_xlim(-75, 75)
        axes[ii].set_ylim(-75, 75)

        axes[ii].set_xticks([])
        axes[ii].set_yticks([])

        axes[-1].plot(alpha, itp_img, color=colors[ii])

    plt.show()

    return 0


def compare_four_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii):
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    triangle_numbers = [int((-1 + np.sqrt(1 + 8 * len(x_p))) / 2) for x_p in x_peaks]

    d_samples_2, x_itp_2, y_itp_2 = get_profile_samples(x_peaks[2][1], y_peaks[2][1], x_peaks[2][-2], y_peaks[2][-2], radii[2])
    d_samples_5, x_itp_5, y_itp_5 = get_profile_samples(x_peaks[5][1], y_peaks[5][1], x_peaks[5][-2], y_peaks[5][-2], radii[5])
    # d_samples_2, x_itp_2, y_itp_2 = get_profile_samples(x_peaks[2][0], y_peaks[2][0], x_peaks[2][-1], y_peaks[2][-1], radii[2])
    # d_samples_5, x_itp_5, y_itp_5 = get_profile_samples(x_peaks[5][0], y_peaks[5][0], x_peaks[5][-1], y_peaks[5][-1], radii[5])
    # d_samples_2, x_itp_2, y_itp_2 = get_profile_samples(x_peaks[2][-1], y_peaks[2][-1], x_peaks[2][-triangle_numbers[2]], y_peaks[2][-triangle_numbers[2]], radii[2])
    # d_samples_5, x_itp_5, y_itp_5 = get_profile_samples(x_peaks[5][-1], y_peaks[5][-1], x_peaks[5][-triangle_numbers[5]], y_peaks[5][-triangle_numbers[5]], radii[5])

    titles = ['Reconstruction', 'Ground truth (GT)', 'Conv. GT (PSF fit)', 'Conv. GT (MTF fit)']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_lim = [np.min(images_list[0]), np.max(images_list[0])]
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 4))

    for ii in range(len(images_list)):
        jj = np.floor(ii / 2).astype(int)
        kk = ii % 2

        interpolator = RegularGridInterpolator((x, y), images_list[ii])
        itp_img_2 = interpolator((x_itp_2, y_itp_2))
        itp_img_5 = interpolator((x_itp_5, y_itp_5))

        axes[jj][kk].imshow(images_list[ii].T, origin='lower', extent=extent, clim=c_lim)
        # axes[jj][kk].plot(x_itp_2, y_itp_2, color=colors[ii])
        # axes[jj][kk].plot(x_itp_5, y_itp_5, color=colors[ii])
        axes[jj][kk].plot([x_itp_2[0], x_itp_2[-1]], [y_itp_2[0], y_itp_2[-1]], color='white')
        axes[jj][kk].plot([x_itp_5[0], x_itp_5[-1]], [y_itp_5[0], y_itp_5[-1]], color='white')
        axes[jj][kk].set_xlim(-70, 70)
        axes[jj][kk].set_ylim(-70, 70)
        axes[jj][kk].set_xticks([])
        axes[jj][kk].set_yticks([])
        axes[jj][kk].set_title(titles[ii], color=colors[ii], fontsize=12)

        axes[0][-1].plot(d_samples_5, itp_img_5, color=colors[ii])
        axes[1][-1].plot(d_samples_2, itp_img_2, color=colors[ii])

    x_ticks = np.arange(-20, 30, 10)

    axes[0][-1].set_xlim(x_ticks[0], x_ticks[-1])
    axes[0][-1].set_ylim(axes[1][-1].get_ylim())
    axes[0][-1].set_xticks(x_ticks)
    axes[0][-1].set_xticklabels([])
    axes[0][-1].ticklabel_format(axis='y', scilimits=(0, 0))
    # axes[0][-1].plot(d_samples_5, np.ones(d_samples_5.shape) * np.max(images_list[1]) * np.pi / (8 * np.sqrt(3)), color='k')

    axes[1][-1].set_xlim(x_ticks[0], x_ticks[-1])
    axes[1][-1].set_xticks(x_ticks)
    x_tick_labels = axes[1][-1].get_xticklabels()
    x_tick_labels[1] = ''
    x_tick_labels[3] = ''
    axes[1][-1].set_xticklabels(x_tick_labels)
    axes[1][-1].ticklabel_format(axis='y', scilimits=(0, 0))

    axes[1][-1].set_xlabel(r'Distance [mm]')

    plt.show()

    return 0


def get_profile_samples(x0, y0, x1, y1, r):
    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    d_samples = np.linspace(-d / 2 - 2 * r, d / 2 + 2 * r, 1000)

    x_itp = x0 * (.5 - d_samples / d) + x1 * (.5 + d_samples / d)
    y_itp = y0 * (.5 - d_samples / d) + y1 * (.5 + d_samples / d)

    return d_samples, x_itp, y_itp


def compare_four_derenzo_images_and_profiles_v2(x_grid, y_grid, images_list, radii, derenzo_contrast_2d):
    titles = ['Reconstruction', 'Ground truth (GT)', 'Conv. GT (PSF fit)', 'Conv. GT (MTF fit)']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_lim = [np.min(images_list[0]), np.max(images_list[0])]
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 4))

    for ii in range(len(images_list)):
        jj = np.floor(ii / 2).astype(int)
        kk = ii % 2

        distances, profiles, x_peaks_itp, y_peaks_itp = derenzo_contrast_2d(images_list[ii], return_profiles=True)

        axes[jj][kk].imshow(images_list[ii].T, origin='lower', extent=extent, clim=c_lim)
        axes[jj][kk].plot(x_peaks_itp[2], y_peaks_itp[2], linewidth=1, color='white')
        axes[jj][kk].plot(x_peaks_itp[5], y_peaks_itp[5], linewidth=1, color='white')
        axes[jj][kk].set_xlim(-70, 70)
        axes[jj][kk].set_ylim(-70, 70)
        axes[jj][kk].set_xticks([])
        axes[jj][kk].set_yticks([])
        axes[jj][kk].set_title(titles[ii], color=colors[ii], fontsize=12)

        axes[jj][kk].text(0, -63, r'$i=3$', color='white', va='center', ha='center', fontsize=10)
        axes[jj][kk].text(0, 61, r'$i=6$', color='white', va='center', ha='center', fontsize=10)

        axes[0][-1].plot(distances[5], np.mean(profiles[5], axis=-1), color=colors[ii])
        axes[1][-1].plot(distances[2], np.mean(profiles[2], axis=-1), color=colors[ii])

    y_lim = [-12, 217]

    axes[0][-1].set_xlim(-2 * radii[5], 2 * radii[5])
    axes[0][-1].set_ylim(y_lim)
    axes[0][-1].set_xticks(np.arange(-2, 3) * radii[5])
    axes[0][-1].set_xticklabels([])
    axes[0][-1].set_yticks([0, 1e2, 2e2])
    axes[0][-1].ticklabel_format(axis='y', scilimits=(0, 0))
    axes[0][-1].plot(distances[5], np.ones(distances[5].shape) * np.max(images_list[1]) * np.pi / (8 * np.sqrt(3)), color='k', linestyle='--')
    axes[0][-1].text(radii[5], 2e2, r'$i=6$', color='black', va='center', ha='left', fontsize=10)
    axes[0][-1].set_title('Hermite-\nGaussian', fontsize=12)
    # axes[0][-1].set_title('Plateau-\npolynomial', fontsize=12)

    axes[1][-1].set_xlim(-2 * radii[2], 2 * radii[2])
    axes[1][-1].set_ylim(y_lim)
    axes[1][-1].set_xticks(np.arange(-2, 3) * radii[2])
    axes[1][-1].set_yticks([0, 1e2, 2e2])
    axes[1][-1].set_xticklabels(['', r'$-R_i$', r'$0$', r'$R_i$', ''])
    axes[1][-1].ticklabel_format(axis='y', scilimits=(0, 0))
    axes[1][-1].text(radii[2], 2e2, r'$i=3$', color='black', va='center', ha='left', fontsize=10)
    # axes[1][-1].set_xlabel(r'Distance [mm]')

    plt.show()

    return 0


if __name__ == "__main__":
    main()
