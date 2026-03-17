"""
Analyze the reconstructions of the Derenzo phantom

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from pickle import load as load_pickle
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Auxiliary functions
from CASToR.read_interfile import read_interfile, accumulate_slices
from point_spread_functions_2d import get_2d_point_spread_function
from mtf_analysis import get_derenzo_mtf, get_sigma
from psf_expansion_validity import alpha_valid, alpha_beta_valid


def main():
    # Load the ideal Derenzo image
    img_peaks = np.load('/Derenzo_phantom/Derenzo_pixelated/img_peaks.npy')
    # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/img_valleys.npy')
    img_valleys = np.load('/Derenzo_phantom/Derenzo_pixelated/img_valleys_parzych.npy')
    x_grid = np.load('/Derenzo_phantom/Derenzo_pixelated/x_grid.npy')
    y_grid = np.load('/Derenzo_phantom/Derenzo_pixelated/y_grid.npy')
    x_out = np.load('/Derenzo_phantom/Derenzo_pixelated/x_out.npy')
    y_out = np.load('/Derenzo_phantom/Derenzo_pixelated/y_out.npy')
    # vis_3d(img_peaks-img_valleys)
    print(x_out.shape)
    sys.exit()

    x_peaks = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/x_peaks.pkl', 'rb'))
    y_peaks = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/y_peaks.pkl', 'rb'))
    r_peaks = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/r_peaks.pkl', 'rb'))

    x_valleys = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/x_valleys.pkl', 'rb'))
    y_valleys = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/y_valleys.pkl', 'rb'))
    r_valleys = load_pickle(open('/Derenzo_phantom/Derenzo_pixelated/r_valleys.pkl', 'rb'))

    radii = np.array([arr[0] for arr in r_peaks])
    # np.save('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/radii.npy', radii)
    img_derenzo = np.sum(img_peaks, axis=-1)

    # pvr = evaluate_peak_to_valley_ratio(x_grid, y_grid, img_derenzo, x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys)
    # pvr_2 = evaluate_peak_to_valley_ratio_v2(img_derenzo, img_peaks, img_valleys)




    # plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.imshow(img_derenzo.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    # ax.set_xlabel(r'$x$ [mm]')
    # ax.set_ylabel(r'$y$ [mm]')
    # plt.show()

    # Load the reconstruction
    gantry = np.array(['tot', 'tbtb', 'tbb', 'bb'])

    sf_data = 1 / (4 * radii)
    mtf_data = np.zeros((6, gantry.size))
    sf_fit = np.zeros((100, gantry.size))
    mtf_fit = np.zeros((100, gantry.size))

    for ii in range(gantry.size):
        output_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo/' + gantry[ii]
        # output_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo_noncol/' + gantry[ii]
        # img_recon_acc = accumulate_slices(output_dir + '/sub_4/img_TB_brain_tot_all_GATE_it*.hdr')
        img_recon_acc = accumulate_slices(output_dir + '/img_TB_brain_%s_all_CONST_it*.hdr' % gantry[ii], idx_1=None)
        img_recon_acc = np.flip(img_recon_acc, axis=1)
        # vis_3d(img_recon_acc)
        # gif_plot(img_recon_acc, sys.path[0] + '/GIFs/1000.gif', step=10)

        """Optimization"""
        # img_convolved_gaussian, fwhm_gaussian, x_1d, psf_gaussian = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'gaussian', visualize=False)
        # compare_derenzo_line_profiles(x_grid, y_grid, img_recon_acc[:, :, 500], img_convolved_gaussian, x_out, y_out, idx=1)

        # img_convolved_bspline, fwhm_bspline, x_1d, psf_bspline = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'interpolation', visualize=False)
        # compare_derenzo_line_profiles(x_grid, y_grid, img_recon_acc[:, :, 500], img_convolved_bspline, x_out, y_out, idx=1)

        # img_convolved_expansion, fwhm_expansion, x_1d, psf_expansion = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'gaussian-expansion', visualize=False)
        img_convolved_expansion, fwhm_expansion, x_1d, psf_expansion = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'bessel', visualize=False)
        # compare_derenzo_line_profiles(x_grid, y_grid, img_recon_acc[:, :, 500], img_convolved_expansion, x_out, y_out, idx=1)

        get_sigma(x_1d, psf_expansion)

        get_fwhm(x_1d, psf_expansion, vis=True)

        sys.exit()

        # # Run for all iterations (in parallel)
        # def process_item(item):
        #     return run_optimization(x_grid, y_grid, img_recon_acc[:, :, item], img_derenzo, 'gaussian', visualize=False, return_only_fwhm=True)
        # fwhm_it = np.array(Parallel(n_jobs=16)(delayed(process_item)(item) for item in trange(img_recon_acc.shape[-1])))
        # np.save(sys.path[0] + '/Parameter_progression/FWHM_gaussian.npy', fwhm_it)
        # show_convergence_iterations()

        # _, _, _, psf_lorentzian = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'lorentzian')

        img_convolved_lorentzian, fwhm_gen_lorentzian, _, psf_gen_lorentzian = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'generalized lorentzian')
        # compare_derenzo_line_profiles(x_grid, y_grid, img_recon_acc[:, :, 500], img_convolved_lorentzian, x_out, y_out, idx=1)

        # _, _, _, psf_gen_center_lorentzian = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'generalized center lorentzian')

        img_convolved_polynomial, fwhm_polynomial, _, psf_polynomial = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'polynomial')
        # compare_derenzo_line_profiles(x_grid, y_grid, img_recon_acc[:, :, 500], img_convolved_polynomial, x_out, y_out, idx=1)

        compare_point_spread_functions(x_1d, [psf_gaussian, psf_gen_lorentzian, psf_polynomial],
                                       [fwhm_gaussian, fwhm_gen_lorentzian, fwhm_polynomial],
                                       ['Gaussian', 'Gen. Lorentzian', 'Polynomial'])

        compare_point_spread_functions(x_1d, [psf_polynomial], [fwhm_polynomial], ['PSF'])

        #
        # _, fwhm_polynomial, _, _ = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo, 'polynomial')
        # print('%s: %1.2f' % (gantry[ii].upper(), fwhm_polynomial))

        """MTF analysis"""

        # pvr = evaluate_peak_to_valley_ratio(x_grid, y_grid, img_recon_acc[:, :, 500], x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys)

        _, mtf_data[:, ii], sf_fit[:, ii], mtf_fit[:, ii] = get_derenzo_mtf(img_recon_acc[:, :, 500], img_peaks, img_valleys, radii)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    for ii in range(gantry.size):
        ax.plot(sf_data, mtf_data[:, ii], 'x', color=colors[ii])
        ax.plot(sf_fit[:, ii], mtf_fit[:, ii], color=colors[ii], label=gantry[ii].upper())
    ax.set_xlim(0, 0.32)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Spatial frequency [1/mm]')
    ax.set_ylabel('Contrast [%]')
    ax.grid()
    ax.legend()
    plt.show()

    return 0



def show_convergence_iterations():

    fwhm_it = np.load(sys.path[0] + '/Parameter_progression/FWHM_gaussian.npy')
    it = np.arange(1, fwhm_it.size + 1)

    plateau = np.polyfit(it[it > 500], fwhm_it[it > 500], 0)[0][0]

    plt.rcParams.update({'font.size': 12})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.plot(it, fwhm_it)
    ax0.plot([it[0], it[-1]], [plateau, plateau], '--')
    ax0.set_xlim(it[0] - 1, it[-1] + 1)
    ax0.set_ylim(2.5, 5)
    # ax.set_yscale('log')
    ax0.set_xlabel('Iteration number')
    ax0.set_ylabel('FWHM [mm]')

    lin_thresh = 0.01

    ax1.plot(it, fwhm_it - plateau)
    ax1.plot([it[0], it[-1]], [0, 0], '--')
    # ax1.set_xscale('log')
    ax1.set_yscale('symlog', linthresh=lin_thresh, linscale=0.5)
    ax1.set_xlim(it[0] - 1, it[-1] + 1)
    ax1.set_xlabel('Iteration number')
    ax1.set_ylabel('FWHM - %1.2f [mm]' % plateau)
    ax1.add_patch(Rectangle((it[0] - 1, -lin_thresh), it[-1], 2 * lin_thresh, edgecolor='none', facecolor='gray', alpha=0.25))
    plt.show()

    return 0


def compare_point_spread_functions(x, psf_list, fwhm_list, label_list):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    for ii in range(len(psf_list)):
        ax.plot(x, psf_list[ii], label=label_list[ii], color=colors[ii])
        psf_max = np.max(psf_list[ii])
        ax.plot([-fwhm_list[ii] / 2, fwhm_list[ii] / 2], [psf_max / 2, psf_max / 2], linestyle='--', color=colors[ii])
        ax.text(0, psf_max / 2, '%1.2f mm' % fwhm_list[ii], ha='center', va='bottom', color=colors[ii])

    ax.set_xlim(-5, 5)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel('Distance [mm]')
    ax.set_ylabel('Central profile')
    ax.legend(loc='lower center')
    # ax.set_title('Point spread function', weight='bold')
    plt.show()

    return 0


def evaluate_peak_to_valley_ratio(x_grid, y_grid, img, x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys):
    # Generate the interpolation coordinates
    x_peaks_itp, y_peaks_itp = get_interpolation_points(x_peaks, y_peaks, r_peaks)
    x_valleys_itp, y_valleys_itp = get_interpolation_points(x_valleys, y_valleys, r_valleys)

    # fig, ax = plt.subplots()
    # ax.imshow(img.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    #
    # for ii in range(len(x_peaks)):
    #     for jj in range(x_peaks[ii].size):
    #         ax.add_artist(Circle((x_peaks[ii][jj], y_peaks[ii][jj]), r_peaks[ii][jj], edgecolor='tab:red', facecolor='none'))
    #
    #     for jj in range(x_valleys[ii].size):
    #         ax.add_artist(Circle((x_valleys[ii][jj], y_valleys[ii][jj]), r_valleys[ii][jj], edgecolor='tab:orange', facecolor='none'))
    #
    #     # ax.scatter(x_peaks_itp[ii], y_peaks_itp[ii], color='tab:red')
    #     # ax.scatter(x_valleys_itp[ii], y_valleys_itp[ii], color='tab:orange')
    #
    # plt.show()

    # Interpolate
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2
    interpolator = RegularGridInterpolator((x, y), img)

    peak_to_valley_ratio = np.zeros(len(x_peaks_itp))

    for ii in range(len(x_peaks_itp)):
        peaks = interpolator((x_peaks_itp[ii], y_peaks_itp[ii]))
        valleys = interpolator((x_valleys_itp[ii], y_valleys_itp[ii]))

        peak_to_valley_ratio[ii] = np.mean(peaks) / np.mean(valleys)
        # peak_to_valley_ratio[ii] = np.median(peaks) / np.median(valleys)

    return peak_to_valley_ratio


def circle_samples(x_0, y_0, r, n, n_gon=6, exclude_edge=True, visualize=False):
    # Radii
    rho = np.arange(n + 1) / n * r
    if exclude_edge:
        rho *= (n - 1 / 2) / n

    # Number of angular elements
    n_elem = np.arange(n + 1) * n_gon
    n_elem[0] = 1

    x, y = [], []
    for ii in range(rho.size):
        ang = np.arange(n_elem[ii]) / n_elem[ii]
        x.append(rho[ii] * np.cos(ang * 2 * np.pi))
        y.append(rho[ii] * np.sin(ang * 2 * np.pi))

    x, y = np.concatenate(x) + x_0, np.concatenate(y) + y_0

    if visualize:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_aspect(1)
        ax.add_artist(Circle(xy=(x_0, y_0), radius=r, facecolor='none', edgecolor='k'))
        ax.set_xlim(x_0 - r, x_0 + r)
        ax.set_ylim(y_0 - r, y_0 + r)
        plt.show()

    return x, y


def get_interpolation_points(x_list, y_list, radii):
    x_list_itp, y_list_itp = [], []

    for ii in range(len(x_list)):
        x_list_temp, y_list_temp = [], []

        for jj in range(x_list[ii].size):
            x_temp, y_temp = circle_samples(x_list[ii][jj], y_list[ii][jj], radii[ii][jj], 3)
            x_list_temp.append(x_temp)
            y_list_temp.append(y_temp)

        x_list_itp.append(np.concatenate(x_list_temp))
        y_list_itp.append(np.concatenate(y_list_temp))

    return x_list_itp, y_list_itp


def get_fwhm(x, y, vis=False):
    h = np.max(y) / 2

    idx_pre, = np.where(((y[:-1] - h) * (y[1:] - h)) < 0)
    x_0, x_1, y_0, y_1 = x[idx_pre], x[idx_pre + 1], y[idx_pre], y[idx_pre + 1]

    m = (y_1 - y_0) / (x_1 - x_0)
    t = (y_0 * x_1 - y_1 * x_0) / (x_1 - x_0)

    x_h = (h - t) / m

    print('FWHM: %1.3f mm' % (x_h[1] - x_h[0]))

    if vis:
        fig, ax = plt.subplots()
        # ax.plot(x, y, 'x')
        # ax.plot(x[idx_pre], y[idx_pre], 'o')
        # ax.plot([x[0], x[-1]], [h, h])
        # ax.plot(x, m[np.newaxis, :] * x[:, np.newaxis] + t[np.newaxis, :])

        ax.plot(x, y)
        ax.plot(x_h, [h, h])

        ax.set_xlim(-10, 10)
        ax.set_ylim(-0.01, 0.1)

        plt.show()

    return 0


if __name__ == "__main__":
    main()
