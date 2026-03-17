"""
Analyze the reconstructions of the Hexagon phantom

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import iqr
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Auxiliary functions
from vis import vis_3d
from read_interfile import read_interfile, accumulate_slices
from derenzo_contrast import continuous_fourier_cosine_transform


def main():
    # Get the Hexagon phantom parameters
    x_rods = np.load('/home/martin/PycharmProjects/J-PET/Hexagon/x_rods.npy', allow_pickle=True)
    y_rods = np.load('/home/martin/PycharmProjects/J-PET/Hexagon/y_rods.npy', allow_pickle=True)
    radii = np.load('/home/martin/PycharmProjects/J-PET/Hexagon/radii.npy')

    img_hexagon = np.load('/home/martin/PycharmProjects/J-PET/Hexagon/img_hexagon.npy')
    img_mask = np.load('/home/martin/PycharmProjects/J-PET/Hexagon/img_mask.npy')

    x_rods, y_rods = np.concatenate(x_rods), np.concatenate(y_rods)
    rho_rods = np.sqrt(x_rods ** 2 + y_rods ** 2)
    phi_rods = np.arctan2(y_rods, x_rods)
    rho_bins = np.linspace(0., 160., 12)
    # rho_bins = np.array([0., 50.])
    rho_mids = (rho_bins[1:] + rho_bins[:-1]) / 2
    rho_idx = np.digitize(rho_rods, bins=rho_bins) - 1
    # print(np.bincount(rho_idx))
    # print(np.sum(np.bincount(rho_idx)))

    # Image properties
    x, y, z, _ = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/Hexagon/TB_6_30_3_BI_4_18_3_TOT_CONST/img_it1.hdr', return_grid=True)
    dx, dy = x[1] - x[0], y[1] - y[0]
    x_grid, y_grid = np.append(x - dx / 2, x[-1] + dx / 2), np.append(y - dy / 2, y[-1] + dy / 2)

    #
    hexagon_contrast = get_hexagon_contrast_function(x_grid, y_grid, x_rods, y_rods, phi_rods, rho_mids, rho_idx, radii)
    hexagon_2 = get_hexagon_2_function(img_hexagon, img_mask, rho_idx)
    # hexagon_contrast(41.75 * np.sum(img_hexagon, axis=-1))


    tb_res = [6, 30, 3]
    bi_res = [4, 18, 3]
    gantry = np.array(['TOT', 'TBTB', 'TBB', 'BB'])
    tof = np.array(['inf_ps', '500_ps', '300_ps', '100_ps'])
    # normalization = 'CONST'
    normalization = 'GATE'
    flt = ''  # 'true'

    contrast_analysis = True
    uniformity_analysis = False
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    for jj in range(0, gantry.size - 0):
        # Load the image
        # output_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/Hexagon'
        output_dir = '/home/martin/J-PET/CASToR_RECONS/RECONS/Hexagon'
        # img_dir = '/TB_%d_%d_%d_BI_%d_%d_%d_%s_%s_%s' % (tb_res[0], tb_res[1], tb_res[2], bi_res[0], bi_res[1], bi_res[2], gantry[jj], tof[0], normalization)
        img_dir = '/TB_%d_%d_%d_BI_%d_%d_%d_%s_%s_%s' % (tb_res[0], tb_res[1], tb_res[2], bi_res[0], bi_res[1], bi_res[2], gantry[3], tof[3], normalization)

        # img_recon_acc = accumulate_slices(output_dir + img_dir + '/img_it*.hdr', idx_1=None)
        # img_recon_acc = np.flip(img_recon_acc, axis=1)
        # vis_3d(img_recon_acc)

        img_recon = read_interfile(output_dir + img_dir + '/img_it500.hdr', return_grid=False)
        # img_recon = np.flip(img_recon, axis=1)
        img_recon, z = np.flip(img_recon, axis=1)[:, :, 15:-15], z[15:-15]
        img_recon_mean = np.mean(img_recon, axis=-1)
        vis_3d(img_recon)

        fig, ax = plt.subplots()
        ax.imshow(img_recon_mean.T, origin='lower')
        plt.show()

        # img_recon_mean = 41.75 * np.sum(img_hexagon, axis=-1)

        """Plot the image"""
        # fig, ax = plt.subplots(figsize=(6, 6))
        # # ax.imshow(img_recon[:, :, 25].T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
        # # ax.imshow(img_recon_mean.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], vmin=0, vmax=np.percentile(img_recon_mean.flatten(), 99.5))
        # im = ax.imshow(img_recon_mean.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], vmin=0, vmax=60)
        # cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im, cax=cax, orientation='vertical')
        # ax.set_xlabel(r'$x$ [mm]')
        # ax.set_ylabel(r'$y$ [mm]')
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(-50, 50)
        # # ax.set_xticks([-200, -100, 0, 100, 200])
        # # ax.set_yticks([-200, -100, 0, 100, 200])
        # plt.show()

        if contrast_analysis:
            rho_mids, mean_contrast_lat, std_contrast_lat, mean_contrast_rad, std_contrast_rad = hexagon_contrast(img_recon_mean)
            # hexagon_contrast(img_recon[:, :, 25])
            # ax.fill_between(rho_mids, y1=mean_contrast_lat + std_contrast_lat, y2=mean_contrast_lat - std_contrast_lat, alpha=0.25, facecolor=colors[jj])
            ax.errorbar(rho_mids, mean_contrast_lat, yerr=std_contrast_lat, fmt='o', alpha=0.25, capsize=2, color=colors[jj])
            # ax.plot(rho_mids, mean_contrast_lat, color=colors[jj], label=gantry[jj].upper())
            ax.plot(rho_mids, mean_contrast_lat, color=colors[jj], label='%s ps' % tof[jj][:3])
            # ax.fill_between(rho_mids, y1=mean_contrast_rad + std_contrast_rad, y2=mean_contrast_rad - std_contrast_rad, alpha=0.25, facecolor=colors[jj], edgecolor='white', hatch='//')
            ax.errorbar(rho_mids, mean_contrast_rad, yerr=std_contrast_rad, fmt='o', alpha=0.25, capsize=2, color=colors[jj])
            ax.plot(rho_mids, mean_contrast_rad, linestyle='--', color=colors[jj])

        if uniformity_analysis:
            residuals_mean, residuals_std = hexagon_2(img_recon_mean)
            ax.stairs(residuals_mean, edges=rho_bins, color=colors[jj], label='%s ps' % tof[jj][:3])
            ax.errorbar(rho_mids, residuals_mean, yerr=residuals_std, color=colors[jj], fmt='none', capsize=2)

    if contrast_analysis:
        first_legend = ax.legend(loc='lower left', ncol=2, frameon=False)
        # Dummy plots for the second legend
        h_l = ax.errorbar(np.nan, np.nan, yerr=np.nan, capsize=2, linestyle='-', color='k')
        h_r = ax.errorbar(np.nan, np.nan, yerr=np.nan, capsize=2, linestyle='--', color='k')
        ax.legend([h_l, h_r], ['Lateral', 'Radial'], ncol=2, frameon=False, loc='upper center')
        ax.add_artist(first_legend)
        ax.set_ylim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel(r'$\rho$ [mm]')
        ax.set_ylabel('Contrast')

    if uniformity_analysis:
        ax.set_xlim(rho_bins[0], rho_bins[-1])
        ax.set_ylim(40, 90)
        ax.set_xlabel(r'$\rho$ [mm]')
        ax.set_ylabel(r'$\left[\sum_i \left(f_i-\alpha\cdot g_i\right)^2\right]^{1/2}$')
        ax.legend(loc='upper center', ncol=2, frameon=False)

    plt.show()

    return 0


def get_hexagon_contrast_function(x_grid, y_grid, x_rods, y_rods, phi_rods, rho_mids, rho_idx, radii):
    x, y = (x_grid[:-1] + x_grid[1:]) / 2, (y_grid[:-1] + y_grid[1:]) / 2

    x_samples = np.linspace(-2 * radii, 2 * radii, num=101)[:, np.newaxis]
    y_samples = np.zeros(x_samples.shape)

    x_itp_lat, y_itp_lat = [], []
    x_itp_rad, y_itp_rad = [], []
    n_rods = []

    for ii in np.unique(rho_idx):
        n_rods.append(np.sum(rho_idx == ii))
        phi_rods_ii = phi_rods[np.newaxis, rho_idx == ii]
        x_rods_ii = x_rods[np.newaxis, rho_idx == ii]
        y_rods_ii = y_rods[np.newaxis, rho_idx == ii]

        x_itp_lat.append(- x_samples * np.sin(phi_rods_ii) - y_samples * np.cos(phi_rods_ii) + x_rods_ii)
        y_itp_lat.append(x_samples * np.cos(phi_rods_ii) - y_samples * np.sin(phi_rods_ii) + y_rods_ii)

        x_itp_rad.append(x_samples * np.cos(phi_rods_ii) - y_samples * np.sin(phi_rods_ii) + x_rods_ii)
        y_itp_rad.append(x_samples * np.sin(phi_rods_ii) + y_samples * np.cos(phi_rods_ii) + y_rods_ii)

    n_rods = np.array(n_rods)
    x_itp_lat, y_itp_lat = np.hstack(x_itp_lat), np.hstack(y_itp_lat)
    x_itp_rad, y_itp_rad = np.hstack(x_itp_rad), np.hstack(y_itp_rad)

    idx_sep = np.cumsum(np.insert(n_rods, 0, 0))

    def hexagon_contrast(img):
        # # Show the interpolation lines
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.imshow(img.T, origin='lower', extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
        # ax.plot(x_itp_lat, y_itp_lat, 'tab:red')
        # ax.plot(x_itp_rad, y_itp_rad, 'tab:orange')
        # # ax.plot(x_itp_lat[0, :], y_itp_lat[0, :], 'x', color='tab:red')
        # # ax.plot(x_itp_rad[0, :], y_itp_rad[0, :], 'x', color='tab:orange')
        # ax.plot(np.nan, 'tab:red', label='Lateral')
        # ax.plot(np.nan, 'tab:orange', label='Radial')
        # ax.set_xlabel(r'$x$ [mm]')
        # ax.set_ylabel(r'$y$ [mm]')
        # # ax.set_xlim(-30, 30)
        # # ax.set_ylim(-30, 30)
        # ax.set_xticks([-200, -100, 0, 100, 200])
        # ax.set_yticks([-200, -100, 0, 100, 200])
        # ax.legend(loc='upper right')
        # plt.show()

        img_itp_lat = RegularGridInterpolator((x, y), img)((x_itp_lat, y_itp_lat))
        img_itp_rad = RegularGridInterpolator((x, y), img)((x_itp_rad, y_itp_rad))

        mean_contrast_lat, std_contrast_lat = np.zeros(n_rods.size), np.zeros(n_rods.size)
        mean_contrast_rad, std_contrast_rad = np.zeros(n_rods.size), np.zeros(n_rods.size)

        ideal = np.zeros(img_itp_lat.shape)
        ideal[(x_samples.flatten() > -2) & (x_samples.flatten() < 2), :] = 1.
        ideal = ideal[:, :2]


        continuous_fourier_cosine_transform(x_samples.flatten(), ideal, n_max=1, show_expansion=True)

        for jj in range(n_rods.size):
            mean_contrast_lat[jj], std_contrast_lat[jj] = continuous_fourier_cosine_transform(x_samples.flatten(), img_itp_lat[:, idx_sep[jj]:idx_sep[jj + 1]], n_max=1, show_expansion=False)
            mean_contrast_rad[jj], std_contrast_rad[jj] = continuous_fourier_cosine_transform(x_samples.flatten(), img_itp_rad[:, idx_sep[jj]:idx_sep[jj + 1]], n_max=1, show_expansion=False)

        return rho_mids, mean_contrast_lat, std_contrast_lat, mean_contrast_rad, std_contrast_rad

    return hexagon_contrast


def get_hexagon_2_function(img_hexagon, img_mask, rho_idx):

    rho_idx_unique = np.unique(rho_idx)

    def hexagon_2(img):
        alphas = np.zeros(img_hexagon.shape[-1])
        residuals = np.zeros(img_hexagon.shape[-1])

        for ii in trange(img_hexagon.shape[-1]):
            img_masked = img[img_mask[:, :, ii]]
            ground_truth_masked = img_hexagon[:, :, ii][img_mask[:, :, ii]]

            alphas[ii] = np.sum(img_masked * ground_truth_masked) / np.sum(ground_truth_masked ** 2)

            # fig, (ax0, ax1) = plt.subplots(1, 2)
            # im = ax0.imshow(img * img_mask[:, :, ii])
            # c_lim = im.get_clim()
            # ax1.imshow(img_hexagon[:, :, ii] * alphas[ii], vmin=c_lim[0], vmax=c_lim[1])
            # plt.show()

            residuals[ii] = np.sqrt(np.sum((img_masked - alphas[ii] * ground_truth_masked) ** 2))

        residuals_means = np.zeros(rho_idx_unique.size)
        residuals_std = np.zeros(rho_idx_unique.size)

        for jj in range(rho_idx_unique.size):
            selection = rho_idx == rho_idx_unique[jj]
            # residuals_means[jj] = np.sum(residuals[selection])
            # residuals_means[jj] = np.sum(residuals[selection]) / np.sum(selection)
            residuals_means[jj] = np.mean(residuals[selection])
            residuals_std[jj] = np.std(residuals[selection])
            # residuals_std[jj] = iqr(residuals[selection])

        return residuals_means, residuals_std

    return hexagon_2


if __name__ == "__main__":
    main()
