"""
Analyze the reconstructions of the Derenzo phantom scaled up by a factor of 3

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt

# Auxiliary functions
from vis import vis_3d
from read_interfile import read_interfile, accumulate_slices
from derenzo_contrast import get_derenzo_contrast_function
from psf_opt_2d import run_optimization, compare_derenzo_line_profiles
from mtf_analysis import get_derenzo_mtf


def main():

    # Load the ideal Derenzo image
    img_peaks = np.load('/Derenzo_phantom/Derenzo_pixelated_3/img_peaks.npy')
    # img_valleys = np.load('/home/martin/PycharmProjects/J-PET/Derenzo_pixelated/img_valleys.npy')
    img_valleys = np.load('/Derenzo_phantom/Derenzo_pixelated_3/img_valleys_parzych.npy')
    x_grid = np.load('/Derenzo_phantom/Derenzo_pixelated_3/x_grid.npy')
    y_grid = np.load('/Derenzo_phantom/Derenzo_pixelated_3/y_grid.npy')
    x_peaks = np.load('/Derenzo_phantom/Derenzo_pixelated_3/x_peaks.npy', allow_pickle=True)
    y_peaks = np.load('/Derenzo_phantom/Derenzo_pixelated_3/y_peaks.npy', allow_pickle=True)
    x_valleys = np.load('/Derenzo_phantom/Derenzo_pixelated_3/x_valleys.npy', allow_pickle=True)
    y_valleys = np.load('/Derenzo_phantom/Derenzo_pixelated_3/y_valleys.npy', allow_pickle=True)
    radii = np.load('/Derenzo_phantom/Derenzo_pixelated_3/radii.npy')

    img_derenzo = np.sum(img_peaks, axis=-1)

    #
    derenzo_sizes = np.array([x_peaks[ii].size for ii in range(len(x_peaks))])
    triangle_numbers = np.rint((np.sqrt(1 + 8 * derenzo_sizes) - 1) / 2).astype(int)
    n_0 = (triangle_numbers * (triangle_numbers - 1) / 2).astype(int)
    x_out = np.array([[x_peaks[ii][n_0[ii]], x_peaks[ii][-1]] for ii in range(len(x_peaks))])
    y_out = np.array([[y_peaks[ii][n_0[ii]], y_peaks[ii][-1]] for ii in range(len(y_peaks))])

    derenzo_contrast, derenzo_contrast_slices = get_derenzo_contrast_function(x_grid, y_grid, x_peaks, y_peaks, x_valleys, y_valleys, radii, show_individual=False)
    print(derenzo_contrast(img_derenzo))

    # vis_3d(img_peaks - img_valleys * 0)

    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2

    # fwhm_ssd = np.array([[3.515406071542665, 3.024862216529229, 3.821603119062926, 3.688664713579763],
    #                      [3.50854072229338, 2.9592188071707457, 3.810146872127501, 3.7139615968343893],
    #                      [3.519717843200914, 2.9754152052677267, 3.817743780527388, 3.7477167705267944]])
    #
    # error_ssd = np.array([[0.07953723669903402, 0.14010009722370406, 0.08000669517106355, 0.11824800705178631],
    #                      [0.06867267312173336, 0.08633136262181489, 0.0605717334848257, 0.055469129929584235],
    #                      [0.062399105726573166, 0.08101527549083301, 0.06770146476377163, 0.0675458766990647]])
    #
    # fwhm_fit = np.array([[3.9315186255690024, 3.4863864955869297, 4.244741800861808, 4.063382106369259],
    #                      [3.919352336838574, 3.425573986027436, 4.237495633944149, 4.061917821058762],
    #                      [3.9193054892676766, 3.467633704118679, 4.229438576133778, 4.0875888303698495]])
    #
    # error_fit = np.array([[0.022313671065237628, 0.036129764003144604, 0.031230826322363292, 0.02913277683969806],
    #                      [0.022097490965792055, 0.030060319362008148, 0.029090779749888718, 0.025037371627994354],
    #                      [0.020844433957743824, 0.02604721788282438, 0.03133007682317057, 0.019785364547093565]])
    #
    # plt.rcParams.update({'font.size': 12})
    # axial_resolution = np.array([3, 6, 10])
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # fig, ax = plt.subplots()
    # for ii in range(gantry.size):
    #     # ax.errorbar(axial_resolution, fwhm_ssd[:, ii], yerr=error_ssd[:, ii], fmt='-', capsize=2, label=gantry[ii].upper(), color=colors[ii])
    #     ax.errorbar(axial_resolution, fwhm_fit[:, ii], yerr=error_fit[:, ii], fmt='-', capsize=2, label=gantry[ii].upper(), color=colors[ii])
    #
    # ax.set_xlabel('Axial resolution [mm]')
    # ax.set_ylabel('FWHM [mm]')
    # ax.legend(ncol=4)
    # plt.show()
    # sys.exit()



    # Image properties
    ii = 0
    tb_res = [6, 30, 3]
    bi_res = [6, 30, 3]
    # bi_res = [4, 18, 3]
    gantry = np.array(['tot', 'tbtb', 'tbb', 'bb'])
    flt = 'all'  # 'true'
    normalization = 'CONST'
    # normalization = 'CONST_DOI'

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(7, 4))

    aa = [3, 6, 10]

    for jj in range(gantry.size):
    # for jj in range(len(aa)):
        # Load the image
        output_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/Derenzo_3'
        img_dir = '/TB_%d_%d_%d_BI_%d_%d_%d_%s_%s_%s' % (tb_res[0], tb_res[1], tb_res[2], bi_res[0], bi_res[1], bi_res[2], gantry[jj], flt, normalization)
        # img_dir = '/TB_%d_%d_%d_BI_%d_%d_%d_%s_%s_%s' % (tb_res[0], tb_res[1], aa[jj], bi_res[0], bi_res[1], aa[jj], gantry[0], flt, normalization)

        # img_recon_acc = accumulate_slices(output_dir + img_dir + '/img_it*.hdr', idx_1=None)
        # img_recon_acc = np.flip(img_recon_acc, axis=1)
        # vis_3d(img_recon_acc)

        img_recon = read_interfile(output_dir + img_dir + '/img_it500.hdr', return_grid=False)
        img_recon = np.flip(img_recon, axis=1)[:, :, 56:104]
        # img_recon = np.flip(img_recon, axis=1)
        # vis_3d(img_recon)

        # ax.plot(np.sum(img_recon, axis=(0, 1)), label='%d mm' % aa[jj])

        # Contrast
        # _, contrast_old, _, _ = get_derenzo_mtf(img_recon_acc[:, :, 500], img_peaks, img_valleys, radii)
        # _, contrast_old, _, _ = get_derenzo_mtf(img_derenzo, img_peaks, img_valleys, radii)

        spatial_frequency = 1 / (4 * radii)
        contrast_slices = derenzo_contrast_slices(img_recon)

        xi = np.linspace(0, 0.5, 100)

        contrast_slices_50 = np.percentile(contrast_slices, 50, axis=-1)
        contrast_slices_25 = np.percentile(contrast_slices, 25, axis=-1)
        contrast_slices_75 = np.percentile(contrast_slices, 75, axis=-1)

        # print(np.vstack((contrast_slices_25[np.newaxis, :], contrast_slices_75[np.newaxis, :])).shape)
        # sys.exit()


        # ax.plot(spatial_frequency, contrast, 'x', color='tab:orange')
        # ax.plot(spatial_frequency, contrast_slices, 'x', color='tab:green')
        # ax.plot(spatial_frequency, np.mean(contrast_slices, axis=1), 'x', color='tab:green')
        # ax.plot(spatial_frequency_slices, contrast_slices.flatten(order='F'), 'x', color='tab:green')
        # ax.plot(spatial_frequency, contrast_old, 'x', color='tab:green')
        h_data = ax.errorbar(spatial_frequency, contrast_slices_50, fmt='none', capsize=2, elinewidth=1,
                    yerr=np.vstack(((contrast_slices_50 - contrast_slices_25)[np.newaxis, :], (contrast_slices_75 - contrast_slices_50)[np.newaxis, :])), color=colors[jj], label=gantry[jj].upper())

        fourier_profiles = np.zeros((xi.size, img_recon.shape[-1]))
        fwhms = np.zeros(img_recon.shape[-1])
        for ii in tqdm(range(img_recon.shape[-1])):
            p_opt, fwhm, objective_function, blur_derenzo, fourier_profile = run_optimization(x_grid, y_grid, img_recon[:, :, ii], img_derenzo,'gaussian-expansion', visualize=False)
            # img_convolved_gauss, fwhm_gen_gauss, _, psf_gen_gauss, fourier_profile, p_opt = run_optimization(x_grid, y_grid, img_recon_acc[:, :, 500], img_derenzo,'bessel', visualize=False)

            fourier_profiles[:, ii] = fourier_profile(xi, *p_opt.x[1:])
            fwhms[ii] = fwhm(p_opt.x[1:])
            # compare_derenzo_line_profiles(x_grid, y_grid, img_recon[:, :, ii], img_convolved_gauss, x_out, y_out, 2)

        # print(np.mean(fwhms))
        # print(np.std(fwhms))
        # sys.exit()

        # ax.plot(xi, fourier_profiles)
        # h_ssd, = ax.plot(xi, np.percentile(fourier_profiles, 50, axis=-1), color=colors[jj])
        # ax.fill_between(xi, np.percentile(fourier_profiles, 25, axis=-1), np.percentile(fourier_profiles, 75, axis=-1), alpha=0.5, color=colors[jj])


        # p_opt2, p_cov2 = curve_fit(fourier_profile, spatial_frequency, contrast, p0=p_opt.x[1:], bounds=([0, -0.1], [np.inf, 1]))
        p_opt2, p_cov2 = curve_fit(fourier_profile, np.tile(spatial_frequency, img_recon.shape[-1]), contrast_slices.flatten(order='F'), p0=p_opt.x[1:], bounds=([0, -0.1], [np.inf, 1]))
        p_err2 = np.sqrt(np.diag(p_cov2))

        df_dx = (fwhm([p_opt2[0] + p_err2[0], p_opt2[1]]) - fwhm(p_opt2)) / p_err2[0]
        df_dy = (fwhm([p_opt2[0], p_opt2[1] + p_err2[1]]) - fwhm(p_opt2)) / p_err2[1]

        print('Objective function 2: %1.2f' % (objective_function([p_opt.x[0], p_opt2[0], p_opt2[1]]) / 1e5))
        # print('FWHM 2: %1.3f' % fwhm(p_opt2))
        print(fwhm(p_opt2))
        print(np.abs(df_dx) * p_err2[0] + np.abs(df_dy) * p_err2[1])

        p_opt, fwhm, objective_function, blur_derenzo, fourier_profile = run_optimization(x_grid, y_grid, np.mean(img_recon, axis=-1), img_derenzo, 'gaussian-expansion', visualize=False)
        compare_derenzo_line_profiles(x_grid, y_grid, np.mean(img_recon, axis=-1), p_opt.x[0] * blur_derenzo(p_opt.x[1:]), x_out, y_out, 4)
        # compare_derenzo_line_profiles(x_grid, y_grid, np.mean(img_recon, axis=-1), p_opt.x[0] * blur_derenzo(p_opt2), x_out, y_out, 4)


        h_fit, = ax.plot(xi, fourier_profile(xi, *p_opt2))
        # plt.show()
    ax.set_xlim(xi[0], xi[-1])
    ax.set_xlabel('Spatial frequency [1/mm]')
    ax.set_ylabel('Normalized contrast')
    # ax.legend([h_ssd, h_data, h_fit], ['Minimized SSD', 'Contrast data', 'Fit on contrast data'])
    ax.legend([h_data, h_fit], ['Contrast data', 'Fit on contrast data'])



    # ax.set_ylim(0, 1.6e5)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_xlim(95, 115)
    # ax.set_xlabel(r'$z$ [mm]')
    # ax.set_ylabel('Laterally integrated image')
    first_legend = ax.legend(loc='upper center', ncol=4)

    fit_dummy, = ax.plot(np.nan, color='k')

    data_dummy = ax.errorbar(np.nan, np.nan, fmt='none', capsize=2, elinewidth=1, yerr=np.nan, color='k')

    ax.legend(handles=[data_dummy, fit_dummy], labels=['Contrast data', 'PSF Fourier fit'], loc='upper right')
    ax.add_artist(first_legend)

    # ax.annotate(r'$6\times30$ mm$^2$' + '\nscintillators\nin brain ins.\n(solid lines)', xy=(0.16, 0.2), xytext=(0.13, 0.2), textcoords='data', va='center', ha='right',
    #             arrowprops=dict(arrowstyle='->', color='black'), fontsize=16, color='black', bbox=None)
    ax.annotate(r'$4\times18$ mm$^2$' + '\nscintillators\nin brain ins.\n(dashed lines)', xy=(0.2, 0.5), xytext=(0.25, 0.5), textcoords='data', va='center', ha='left',
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=16, color='black', bbox=None)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
