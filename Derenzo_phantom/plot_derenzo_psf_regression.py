"""
Plot the convergence of the Point Spread Function (PSF) regression

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Auxiliary functions
from Derenzo_phantom.get_derenzo_image import get_ground_truth_derenzo_image
from Derenzo_phantom.get_derenzo_contrast import get_derenzo_contrast_function
from Derenzo_phantom.get_derenzo_image import load_derenzo_image
from Derenzo_phantom.psf_mtf_regression import fit_psf_2d, blur_ground_truth


def main():
    # Load
    _, _, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks = get_ground_truth_derenzo_image()
    img_derenzo = np.sum(img_peaks, axis=-1)
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_true')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/ALL_energy')

    idx_it_200, = np.where(iterations == 200)[0]
    img_recon = np.mean(imgs[:, :, :, idx_it_200], axis=-1)

    # fig, ax = plt.subplots()
    # ax.imshow(img_recon.T, origin='lower')
    # # ax.imshow(img_peaks.T, origin='lower')
    # plt.show()

    model = 'hermite-gaussian'
    # model = 'plateau-polynomial'

    p_opt = fit_psf_2d(x, y, img_recon, img_derenzo, model=model, show_convergence=False)

    # [1.60176445 1.]

    images_list = [img_recon,
                   1.1330291520718943 * p_opt.x[0] * blur_ground_truth(x, y, img_derenzo, model, p_opt.x[1:]),
                   p_opt.x[0] * blur_ground_truth(x, y, img_derenzo, model, [1.60176445, 1.])]  # p_opt.x[0] * img_derenzo
    compare_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii, idx=5)

    sys.exit()




    img_derenzo_convolved = p_opt.x[0] * blur_ground_truth(x, y, img_derenzo, model, p_opt.x[1:], n=0)
    fff = p_opt.x[0] * blur_ground_truth(x, y, img_derenzo, model, [1.60176445,  1.])
    # img_derenzo_convolved = p_opt.x[0] * img_derenzo
    # psf_2d, x_1d, psf_1d = psf(x, y, p_opt.x[1:])

    # fig, ax = plt.subplots()
    # ax.plot(x_grid[1:-1], psf_2d[79, :])
    # ax.plot(x_1d, psf_1d)
    # plt.show()

    c_lim = [np.min(img_recon), np.max(img_recon)]
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    plt.rcParams.update({'font.size': 12})
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 6))
    ax0.imshow(img_recon.T, origin='lower', extent=extent, clim=c_lim)
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    ax0.set_title('Reconstruction')
    ax1.imshow(img_derenzo_convolved.T, origin='lower', extent=extent, clim=c_lim)
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    ax1.set_title('Convolved ground truth')
    # ax2.imshow(p_opt.x[0] * img_derenzo.T, origin='lower', extent=extent, clim=c_lim)
    ax2.imshow(fff.T, origin='lower', extent=extent, clim=c_lim)
    ax2.set_xlabel(r'$x$ [mm]')
    # ax2.set_ylabel(r'$y$ [mm]')
    ax2.set_title('Ground truth')
    plt.show()

    return 0


def compare_derenzo_images_and_profiles(x_grid, y_grid, images_list, x_peaks, y_peaks, radii, idx=0):
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    triangle_numbers = [int((-1 + np.sqrt(1 + 8 * len(x_p))) / 2) for x_p in x_peaks]

    # x_0, y_0 = x_peaks[idx][-1], y_peaks[idx][-1]
    x_0, y_0 = x_peaks[idx][0], y_peaks[idx][0]
    x_1, y_1 = x_peaks[idx][-triangle_numbers[idx]], y_peaks[idx][-triangle_numbers[idx]]

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
        axes[ii].plot(x_itp, y_itp, color=colors[ii])

        axes[ii].set_xticks([-50, 0, 50])
        axes[ii].set_yticks([-50, 0, 50])

        axes[-1].plot(alpha, itp_img, color=colors[ii])

    plt.show()

    return 0


def compare_derenzo_line_profiles(x_grid, y_grid, img_0, img_1, x_out, y_out, idx):
    # Use them as midpoint and directions
    x_mid = ((x_out[:, 0] + x_out[:, 1]) / 2)[np.newaxis, :]
    y_mid = ((y_out[:, 0] + y_out[:, 1]) / 2)[np.newaxis, :]

    x_vec = (x_out[:, 1] - x_out[:, 0])[np.newaxis, :]
    y_vec = (y_out[:, 1] - y_out[:, 0])[np.newaxis, :]
    norm = np.sqrt(x_vec ** 2 + y_vec ** 2)
    x_vec, y_vec = x_vec / norm, y_vec / norm

    # Interpolation points
    # t = np.linspace(-18, 18, 100)[:, np.newaxis]
    t = np.linspace(-26, 26, 100)[:, np.newaxis]

    x_itp = (x_mid + t * x_vec)[:, idx]
    y_itp = (y_mid + t * y_vec)[:, idx]
    points = np.stack((x_itp.flatten(), y_itp.flatten()), axis=1)

    # Run the interpolation
    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2
    interpolator_0 = RegularGridInterpolator((x, y), img_0)
    itp_img_0 = interpolator_0(points).reshape(x_itp.shape)

    interpolator_1 = RegularGridInterpolator((x, y), img_1)
    itp_img_1 = interpolator_1(points).reshape(x_itp.shape)

    # Plot
    plt.rcParams.update({'font.size': 16})
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 6))
    fig = plt.figure(figsize=(8, 6.5))
    gs = GridSpec(2, 2, height_ratios=[(1 + np.sqrt(5))/2, 1])
    ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    ax0.imshow(img_0.T, origin='lower', extent=extent)
    ax0.plot(x_itp, y_itp, '--', color='tab:blue')
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    # ax0.set_title('Reconstructed image')
    ax0.set_title(r'$6\times30$ mm$^2$')
    # ax0.set_xticks([-80, -40, 0, 40, 80])
    # ax0.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])

    ax1.imshow(img_1.T, origin='lower', extent=extent, vmin=np.min(img_0), vmax=np.max(img_0))
    # ax1.scatter(x_out, y_out, color='tab:red')
    # ax1.scatter(x_mid, y_mid, color='tab:red')
    # ax1.quiver(x_mid, y_mid, x_vec, y_vec, scale=1/18, scale_units='xy', color='tab:red')
    ax1.plot(x_itp, y_itp, '--', color='tab:orange')
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    ax1.set_yticks([])
    # ax1.set_title('Convolved ground truth')
    ax1.set_title(r'$4\times18$ mm$^2$')


    ax2.plot(t, itp_img_0, color='tab:blue', label='Reconstruction')
    ax2.plot(t, itp_img_1, color='tab:orange', linestyle='--', label='Convolved ground truth')
    # f = 1 / (4 * 2)
    # ax2.plot(t, 1.5e2 * (1 - np.cos(2 * np.pi * f * t)) / 2, color='tab:green')
    # ax2.set_ylim(np.min(img_0), np.max(img_0))
    ax2.set_ylim(-0.05e2, 1.05e2)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax2.set_aspect(np.diff(ax2.get_xlim()) / np.diff(ax2.get_ylim()))
    # ax2.set_xlabel('Distance [mm]')
    ax2.set_xlabel('Profile [mm]')
    ax2.set_ylabel('Intensity [arb. units]')
    ax2.set_title('Profile')
    ax2.legend(ncol=2, frameon=False)

    plt.show()
    return 0


if __name__ == "__main__":
    main()
