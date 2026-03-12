"""
Analyze the noise of reconstructions from CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, LinearLocator

# Auxiliary functions
from read_interfile import read_interfile
from FWHM_fits import profile_fits
# from analyze_recon import profile_fit_comparison
from vis import vis_3d


def main():
    # Path of images
    directory = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/iterations'

    # Get the common coordinate system
    x, y, z, _ = read_interfile(directory + '/img_TB_brain_true_GATE_it1.hdr', return_grid=True)
    z_include = np.zeros(z.shape, dtype=bool)
    z_include[int(z.size / 2)] = True

    #
    z_values = np.array([-999.5, 999.5], ndmin=2)  # mm
    z_indices = np.argmin(np.abs(z[:, np.newaxis] - z_values), axis=0)

    rec_iter = 100

    abc = np.zeros(rec_iter)
    efg = np.zeros(rec_iter)

    for ii in range(1, rec_iter):
        # print(ii + 1)
        hdr_path = directory + '/img_TB_brain_true_GATE_it%d.hdr' % (ii + 1)
        img = read_interfile(hdr_path)

        # z_profile = img[10, 10, :]
        z_profile = np.sum(img, axis=(0, 1))

        # fig, ax = plt.subplots()
        # ax.plot(z, z_profile)
        # plt.show()

        efg[ii] = np.sqrt(np.sum(np.diff(z_profile) ** 2))


        # vis_3d(img)

        fwhm, _, _ = profile_fits(x, y, img, fit_function='lorentzian', z_include=z_include)
        abc[ii] = fwhm[z_include]

        # profile_fit_comparison(x, y, img[:, :, 1215], scale='lin')

        # fig, ax = plt.subplots()
        # ax.plot(x, np.sum(img[:, :, 1215], axis=0))
        # ax.set_yscale('log')
        # plt.show()

        # z_profile = np.sum(img, axis=(0, 1))
        # abc[ii] = np.sqrt(np.sum(np.diff(z_profile[z_indices[0]:z_indices[1] + 1]) ** 2))

        # fig, ax = plt.subplots()
        # ax.plot(z, z_profile)
        # ax.set_ylim(0, 0.1)
        # plt.show()

    num_it = np.arange(1, rec_iter + 1)

    #
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.plot(num_it, abc, linewidth=3)
    # y_lim = [0, 14]
    y_lim = [0, 5]
    ax.plot([4, 4], y_lim, linestyle='--', color='k')
    ax.set_ylim(y_lim)

    x_lim = [2, rec_iter]
    ax.plot(x_lim, [1.25, 1.25], linestyle='--', color='k')
    ax.set_xlim(x_lim)
    ax.set_xscale('log')
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('FWHM [mm]', weight='bold', color='tab:blue')
    ax.set_yticklabels([0, 1, 2, 3, 4, 5], fontweight='bold', color='tab:blue')
    ax.tick_params(axis='y', colors='tab:blue')  # Y-axis ticks

    ax_twin = ax.twinx()
    ax_twin.plot(num_it, efg, linewidth=3, color='tab:orange')
    ax_twin.set_ylim(0, 0.5)
    ax_twin.set_yticklabels([0, 1, 2, 3, 4, 5], fontweight='bold', color='tab:orange')
    # ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_twin.set_ylabel('SSD', weight='bold', color='tab:orange')
    ax_twin.spines['right'].set_color('tab:orange')
    ax_twin.spines['left'].set_color('tab:blue')
    ax_twin.tick_params(axis='y', colors='tab:orange')  # Y-axis ticks

    plt.show()

    return 0


if __name__ == "__main__":
    main()
