"""

"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from theoretical_expectation_inside_outside import load_thnd


def main():
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'

    _, _, d_2l_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_2L_distribution.root')
    _, _, d_2l_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_2L_distribution.root')
    bin_edges_2l, bin_centers_2l, d_2l_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_2L_distribution.root')

    _, _, d_fwhm_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_FWHM_distribution.root')
    _, _, d_fwhm_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_FWHM_distribution.root')
    bin_edges_fwhm, bin_centers_fwhm, d_fwhm_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_FWHM_distribution.root')

    _, _, d_fwhm_squared_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_FWHM_squared_distribution.root')
    _, _, d_fwhm_squared_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_FWHM_squared_distribution.root')
    bin_edges_fwhm_squared, bin_centers_fwhm_squared, d_fwhm_squared_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_FWHM_squared_distribution.root')

    # print(bin_edges_fwhm)
    # print(bin_centers_fwhm)

    # Normalize
    d_2l_all = d_2l_tbtb + d_2l_tbbi + d_2l_bibi
    n_2l_all = np.trapezoid(d_2l_all, x=bin_centers_2l)
    d_2l_all, d_2l_tbtb, d_2l_tbbi, d_2l_bibi = d_2l_all / n_2l_all, d_2l_tbtb / n_2l_all, d_2l_tbbi / n_2l_all, d_2l_bibi / n_2l_all

    d_fwhm_all = d_fwhm_tbtb + d_fwhm_tbbi + d_fwhm_bibi
    n_fwhm_all = np.trapezoid(d_fwhm_all, x=bin_centers_fwhm)
    d_fwhm_all, d_fwhm_tbtb, d_fwhm_tbbi, d_fwhm_bibi = d_fwhm_all / n_fwhm_all, d_fwhm_tbtb / n_fwhm_all, d_fwhm_tbbi / n_fwhm_all, d_fwhm_bibi / n_fwhm_all

    d_fwhm_squared_all = d_fwhm_squared_tbtb + d_fwhm_squared_tbbi + d_fwhm_squared_bibi
    n_fwhm_squared_all = np.trapezoid(d_fwhm_squared_all, x=bin_centers_fwhm_squared)
    d_fwhm_squared_all, d_fwhm_squared_tbtb, d_fwhm_squared_tbbi, d_fwhm_squared_bibi = d_fwhm_squared_all / n_fwhm_squared_all, d_fwhm_squared_tbtb / n_fwhm_squared_all, d_fwhm_squared_tbbi / n_fwhm_squared_all, d_fwhm_squared_bibi / n_fwhm_squared_all

    #
    print(np.trapezoid(d_fwhm_tbtb * bin_centers_fwhm, x=bin_centers_fwhm) / np.trapezoid(d_fwhm_tbtb, x=bin_centers_fwhm))
    print(np.trapezoid(d_fwhm_tbbi * bin_centers_fwhm, x=bin_centers_fwhm) / np.trapezoid(d_fwhm_tbbi, x=bin_centers_fwhm))
    print(np.trapezoid(d_fwhm_bibi * bin_centers_fwhm, x=bin_centers_fwhm) / np.trapezoid(d_fwhm_bibi, x=bin_centers_fwhm))

    #
    print()
    print(np.sqrt(np.trapezoid(d_fwhm_squared_tbtb * bin_centers_fwhm_squared, x=bin_centers_fwhm_squared) / np.trapezoid(d_fwhm_squared_tbtb, x=bin_centers_fwhm_squared)))
    print(np.sqrt(np.trapezoid(d_fwhm_squared_tbbi * bin_centers_fwhm_squared, x=bin_centers_fwhm_squared) / np.trapezoid(d_fwhm_squared_tbbi, x=bin_centers_fwhm_squared)))
    print(np.sqrt(np.trapezoid(d_fwhm_squared_bibi * bin_centers_fwhm_squared, x=bin_centers_fwhm_squared) / np.trapezoid(d_fwhm_squared_bibi, x=bin_centers_fwhm_squared)))

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    # ax0.stairs(d_2l_all, edges=bin_edges_2l, color='tab:blue', label='ALL')
    ax0.stairs(d_2l_tbtb, edges=bin_edges_2l, color='tab:orange', label='TB-TB')
    ax0.stairs(d_2l_tbbi, edges=bin_edges_2l, color='tab:green', label='TB-BI')
    ax0.stairs(d_2l_bibi, edges=bin_edges_2l, color='tab:red', label='BI-BI')
    ax0.set_xlim(bin_edges_2l[0], bin_edges_2l[-1])
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax0.set_xlabel(r'$2L$ [mm]')
    ax0.set_ylabel('PDF')
    ax0.legend(frameon=False)

    # ax1.stairs(d_fwhm_all, edges=bin_edges_fwhm, color='tab:blue', label='All')
    ax1.stairs(d_fwhm_tbtb, edges=bin_edges_fwhm, color='tab:orange', label='TB-TB')
    ax1.stairs(d_fwhm_tbbi, edges=bin_edges_fwhm, color='tab:green', label='TB-BI')
    ax1.stairs(d_fwhm_bibi, edges=bin_edges_fwhm, color='tab:red', label='BI-BI')
    ax1.set_xlim(bin_edges_fwhm[0], bin_edges_fwhm[-1])
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('FWHM [mm]')
    x0, y0, w, h = ax1.get_position().bounds

    w_insert = 0.14
    h_insert = w_insert / w * h
    x0_insert = x0 + w - w_insert - 0.015
    y0_insert = y0 + h - h_insert - 0.015

    ax1_insert = fig.add_axes([x0_insert, y0_insert, w_insert, h_insert])
    ax1_insert.stairs(d_fwhm_squared_tbtb, edges=bin_edges_fwhm_squared, color='tab:orange', label='TB-TB')
    ax1_insert.stairs(d_fwhm_squared_tbbi, edges=bin_edges_fwhm_squared, color='tab:green', label='TB-BI')
    ax1_insert.stairs(d_fwhm_squared_bibi, edges=bin_edges_fwhm_squared, color='tab:red', label='BI-BI')
    # ax1_insert.set_xlim(bin_edges_fwhm_squared[0], bin_edges_fwhm_squared[-1])
    ax1_insert.set_xlim(bin_edges_fwhm_squared[0], 10)
    ax1_insert.set_ylim(0, 0.7)
    # ax1_insert.set_xticks([0, 4, 8, 12])
    ax1_insert.set_yticks([0.2, 0.5])
    ax1_insert.set_xlabel(r'FWHM$^2$ [mm$^2$]')

    plt.show()

    return 0


if __name__ == '__main__':
    main()
