"""
Analyze the pre-computed FWHMs

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions


def main():
    analyze_fwhm_sipm_6mm_depth_30mm()
    analyze_fwhm_sipm_4mm_depth_18mm()

    return 0


def analyze_fwhm_sipm_6mm_depth_30mm():
    # Load the FWHMs
    fwhm_path = '//home/martin/PycharmProjects/J-PET_Python_tools/Line_source_resolution/FWHMs/SiPM_6mm_depth_30mm_line_source'
    z = np.load(fwhm_path + '/z.npy')
    fwhm_tot_all_gauss = np.load(fwhm_path + '/fwhm_tot_all_gauss.npy')
    fwhm_tot_true_gauss = np.load(fwhm_path + '/fwhm_tot_true_gauss.npy')
    fwhm_tbtb_true_gauss = np.load(fwhm_path + '/fwhm_tbtb_true_gauss.npy')
    fwhm_bb_true_gauss = np.load(fwhm_path + '/fwhm_bb_true_gauss.npy')
    fwhm_tbb_true_gauss = np.load(fwhm_path + '/fwhm_tbb_true_gauss.npy')

    fwhm_tot_all_lorentz = np.load(fwhm_path + '/fwhm_tot_all_lorentz.npy')
    fwhm_tot_true_lorentz = np.load(fwhm_path + '/fwhm_tot_true_lorentz.npy')
    fwhm_tbtb_true_lorentz = np.load(fwhm_path + '/fwhm_tbtb_true_lorentz.npy')
    fwhm_bb_true_lorentz = np.load(fwhm_path + '/fwhm_bb_true_lorentz.npy')
    fwhm_tbb_true_lorentz = np.load(fwhm_path + '/fwhm_tbb_true_lorentz.npy')

    fwhm_tot_all_genlorentz = np.load(fwhm_path + '/fwhm_tot_all_genlorentz.npy')
    fwhm_tot_true_genlorentz = np.load(fwhm_path + '/fwhm_tot_true_genlorentz.npy')
    fwhm_tbtb_true_genlorentz = np.load(fwhm_path + '/fwhm_tbtb_true_genlorentz.npy')
    fwhm_bb_true_genlorentz = np.load(fwhm_path + '/fwhm_bb_true_genlorentz.npy')
    fwhm_tbb_true_genlorentz = np.load(fwhm_path + '/fwhm_tbb_true_genlorentz.npy')

    # compare_fwhm(z, [fwhm_tot_all_gauss, fwhm_tot_true_gauss], ['All', 'True'])
    compare_fwhm(z, [fwhm_tot_all_lorentz, fwhm_tot_true_lorentz], ['All coincidences', 'True coincidences'])
    # # compare_fwhm(z, [fwhm_tot_all_genlorentz, fwhm_tot_true_genlorentz], ['All', 'True'])

    # compare_fwhm(z, [fwhm_tot_true_gauss, fwhm_tbtb_true_gauss, fwhm_bb_true_gauss, fwhm_tbb_true_gauss], ['TOT', 'TB-TB', 'B-B', 'TB-B'])
    compare_fwhm(z, [fwhm_tot_true_lorentz, fwhm_tbtb_true_lorentz, fwhm_tbb_true_lorentz, fwhm_bb_true_lorentz], ['TOT', 'TB-TB', 'TB-B', 'B-B'])
    # # compare_fwhm(z, [fwhm_tot_true_genlorentz, fwhm_tbtb_true_genlorentz, fwhm_bb_true_genlorentz, fwhm_tbb_true_genlorentz], ['TOT', 'TB-TB', 'B-B', 'TB-B'])

    # Load the weights
    w_tot_true = np.load(fwhm_path + '/w_tot_true.npy')
    w_tbtb_true = np.load(fwhm_path + '/w_tbtb_true.npy')
    w_bb_true = np.load(fwhm_path + '/w_bb_true.npy')
    w_tbb_true = np.load(fwhm_path + '/w_tbb_true.npy')

    # compare_fwhm(z, [w_tot_true, w_tbtb_true, w_bb_true, w_tbb_true],['TOT', 'TB-TB', 'B-B', 'TB-B'])

    # weighted_sum_fwhm(z, [fwhm_tot_true_gauss, fwhm_tbtb_true_gauss, fwhm_bb_true_gauss, fwhm_tbb_true_gauss],
    #                   [w_tot_true, w_tbtb_true, w_bb_true, w_tbb_true],
    #                   ['TOT', 'TB-TB', 'B-B', 'TB-B'])

    weighted_sum_fwhm(z, [fwhm_tot_true_lorentz, fwhm_tbtb_true_lorentz, fwhm_bb_true_lorentz, fwhm_tbb_true_lorentz],
                      [w_tot_true, w_tbtb_true, w_bb_true, w_tbb_true],
                      ['TOT', 'TB-TB', 'B-B', 'TB-B'])

    # contribution_analysis(z, [fwhm_tot_true_lorentz, fwhm_tbtb_true_lorentz, fwhm_bb_true_lorentz, fwhm_tbb_true_lorentz],
    #                   [w_tot_true, w_tbtb_true, w_bb_true, w_tbb_true],
    #                   ['TOT', 'TB-TB', 'B-B', 'TB-B'])
    return 0


def analyze_fwhm_sipm_4mm_depth_18mm():
    fwhm_path = '//home/martin/PycharmProjects/J-PET_Python_tools/Line_source_resolution/FWHMs/SiPM_4mm_depth_18mm_line_source'
    z = np.load(fwhm_path + '/z.npy')
    fwhm_tot_all_lorentz = np.load(fwhm_path + '/fwhm_tot_all_lorentz.npy')
    fwhm_tot_true_lorentz = np.load(fwhm_path + '/fwhm_tot_true_lorentz.npy')
    fwhm_tbtb_true_lorentz = np.load(fwhm_path + '/fwhm_tbtb_true_lorentz.npy')
    fwhm_tbb_true_lorentz = np.load(fwhm_path + '/fwhm_tbb_true_lorentz.npy')
    fwhm_bb_true_lorentz = np.load(fwhm_path + '/fwhm_bb_true_lorentz.npy')

    # Load the weights
    w_tot_true = np.load(fwhm_path + '/w_tot_true.npy')
    w_tbtb_true = np.load(fwhm_path + '/w_tbtb_true.npy')
    w_bb_true = np.load(fwhm_path + '/w_bb_true.npy')
    w_tbb_true = np.load(fwhm_path + '/w_tbb_true.npy')

    compare_fwhm(z, [fwhm_tot_all_lorentz, fwhm_tot_true_lorentz], ['All coincidences', 'True coincidences'])
    compare_fwhm(z, [fwhm_tot_true_lorentz, fwhm_tbtb_true_lorentz, fwhm_tbb_true_lorentz, fwhm_bb_true_lorentz], ['TOT', 'TB-TB', 'TB-B', 'B-B'])

    # weighted_sum_fwhm(z, [fwhm_tot_true_lorentz, fwhm_tbtb_true_lorentz, fwhm_tbb_true_lorentz, fwhm_bb_true_lorentz],
    #                   [w_tot_true, w_tbtb_true, w_tbb_true, w_bb_true],
    #                   ['TOT', 'TB-TB', 'TB-B', 'B-B'])

    return 0


def compare_fwhm(z, fwhm_list, label_list, n_mvgavg=51, ax=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colors = [colors[0], colors[7], colors[2], colors[3], colors[1]]
    ax_generated = False
    if ax is None:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        ax_generated = True

    # Data
    for ii in range(len(fwhm_list)):
        ax.plot(z, fwhm_list[ii], alpha=0.25, color=colors[ii])

    # Moving average
    for ii in range(len(fwhm_list)):
        fwhm_mvgavg = np.convolve(fwhm_list[ii], np.ones(n_mvgavg) / n_mvgavg, mode='same')
        ax.plot(z, fwhm_mvgavg, color=colors[ii], label=label_list[ii])

    if ax_generated:
        ax.set_ylim(0, 5)
        # ax.legend(loc='lower center')
        ax.legend(loc='lower center')
        ax.set_xlabel(r'$z$ [mm]')
        ax.set_ylabel('FWHM [mm]')
        plt.show()

    return 0


def weighted_sum_fwhm(z, fwhm_list, weight_list, label_list):
    # Pre-processing
    weights = np.vstack(weight_list)
    weights /= np.sum(weights[1:, :], axis=0)

    fwhms = np.vstack(fwhm_list)
    fwhms[fwhms == 0] = np.nan

    z_values = np.array([-1150, 1150], ndmin=2)  # [mm]
    idx_0, idx_1 = np.argmin(np.abs(z[:, np.newaxis] - z_values), axis=0)
    p_values = np.linspace(-4, 2, 51)

    superpositions = []
    errors = np.zeros(p_values.size)

    for ii in range(len(p_values)):
        temp = np.nansum(weights[1:, :] * (fwhms[1:, :] ** p_values[ii]), axis=0)
        temp[temp == 0] = np.nan
        temp = temp ** (1 / p_values[ii])
        temp[np.isnan(temp)] = 0
        superpositions.append(temp)

        errors[ii] = np.sum((fwhm_list[0][idx_0:idx_1] - temp[idx_0:idx_1]) ** 2) ** (1 / 2)


    p_values_show = np.array([2, 1, -1], ndmin=2)
    idx_p_2, idx_p_1, idx_p_m1 = np.argmin(np.abs(p_values[:, np.newaxis] - p_values_show), axis=0)
    first_guess = np.sqrt(((weights[1, :] * fwhm_list[1]) ** 2 + (weights[2, :] * fwhm_list[2]) ** 2 + (weights[3, :] * fwhm_list[3]) ** 2))
    # second_guess = np.sqrt(((weight_list[1] * fwhm_list[1]) ** 2 + (weight_list[2] * fwhm_list[2]) ** 2 + (weight_list[3] * fwhm_list[3]) ** 2)) / np.sqrt(weight_list[1] ** 2 + weight_list[2] ** 2 + weight_list[3] ** 2)
    fwhm_plot = [fwhm_list[0], first_guess, superpositions[idx_p_2], superpositions[idx_p_1], superpositions[idx_p_m1]]
    label_plot = [None,
                  r'$\left(\sum\,w_i^2\sigma_i^2\right)^{1/2}$',
                  r'$\left(\sum\,w_i\sigma_i^2\right)^{1/2}$',
                  r'$\sum\,w_i\sigma_i$',
                  r'$\left(\sum\,w_i\sigma_i^{-1}\right)^{-1}$']

    p_0, e_0 = find_minimum(p_values, errors, show_plot=False)

    plt.rcParams.update({'font.size': 16})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))

    compare_fwhm(z, fwhm_list, label_list, ax=ax0)
    # ax0.set_ylim(0, 5)
    ax0.set_ylim(0, 6)
    ax0.set_xlabel(r'$z$ [mm]')
    ax0.set_ylabel('FWHM [mm]')
    ax0.legend(ncol=2)
    ax0.set_title(r'$\sigma_i$')

    compare_fwhm(z, weight_list, label_list, ax=ax1)
    ax1.set_xlabel(r'$z$ [mm]')
    ax1.set_title(r'$w_i$')
    ax1.legend(ncol=2)

    ax2.plot(p_values, errors)
    ax2.plot(p_0, e_0, 'x', color='black')
    ax2.text(p_0, e_0 * 1.01, '%1.2f' % p_0, ha='center', va='bottom')
    ax2.set_xticks(np.arange(-4, 3))
    ax2.set_xlabel(r'$p$')
    ax2.set_ylabel('RMSE')

    compare_fwhm(z, fwhm_plot, label_plot, ax=ax3)
    # ax3.set_xlim(right=0)
    # ax3.set_ylim(2, 4)
    # ax3.set_ylim(3, 5)
    ax3.set_ylim(2, 5)
    ax3.set_xlabel(r'$z$ [mm]')
    ax3.set_ylabel('FWHM [mm]')
    ax3.legend(ncol=2)

    plt.show()

    return 0


def find_minimum(x, y, show_plot=False):

    xd = (x[1:] + x[:-1]) / 2
    dy = np.diff(y) / np.diff(x)

    x_0 = np.interp(0, dy, xd)
    y_0 = np.interp(x_0, x, y)

    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x_0, y_0, 'x')
        ax_twin = ax.twinx()
        ax_twin.plot(xd, dy, color='tab:orange')
        plt.show()

    return x_0, y_0


def contribution_analysis(z, fwhm_list, weight_list, label_list):

    # weights = np.vstack(weight_list[1:])
    weights = np.vstack([moving_average(weight_list[1]), moving_average(weight_list[2]), moving_average(weight_list[3])])
    weights /= np.sum(weights, axis=0)

    # fwhms = np.vstack(fwhm_list[1:])
    fwhms = np.vstack([moving_average(fwhm_list[1]), moving_average(fwhm_list[2]), moving_average(fwhm_list[3])])
    fwhms[fwhms == 0] = np.nan

    fwhm_inv_tot = np.nansum(weights / fwhms, axis=0)
    fwhm_tot = np.zeros(fwhm_inv_tot.shape)
    fwhm_tot[fwhm_inv_tot != 0] = 1 / fwhm_inv_tot[fwhm_inv_tot != 0]

    w_inv_0 = weights[0, :] / fwhms[0, :] * fwhm_tot
    w_inv_1 = weights[1, :] / fwhms[1, :] * fwhm_tot
    w_inv_2 = weights[2, :] / fwhms[2, :] * fwhm_tot

    # todo: First smooth then normalize

    fig, ax = plt.subplots()
    ax.plot(z, weights[0, :], color='tab:orange')
    ax.plot(z, w_inv_0, linestyle='--', color='tab:orange')

    ax.plot(z, weights[1, :], color='tab:green')
    ax.plot(z, w_inv_1, linestyle='--', color='tab:green')

    ax.plot(z, weights[2, :], color='tab:red')
    ax.plot(z, w_inv_2, linestyle='--', color='tab:red')
    plt.show()


    fig, ax = plt.subplots()
    compare_fwhm(z, [fwhm_list[0], fwhm_tot], [label_list[0], ''], ax=ax)
    ax.set_ylim(0, 6)
    plt.show()

    return 0


def moving_average(x, n_window=51):
    return np.convolve(x, np.ones(n_window) / n_window, mode='same')


if __name__ == "__main__":
    main()
