"""

"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Auxiliary functions
from theoretical_expectation_inside_outside import load_thnd


def main():
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'
    _, _, d_2l_p_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_2L_p_distribution.root')
    _, _, d_2l_p_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_2L_p_distribution.root')
    bin_edges, bin_centers, d_2l_p_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_2L_p_distribution.root')

    _, _, d_fwhm_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_FWHM_distribution.root')
    _, _, d_fwhm_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_FWHM_distribution.root')
    bin_edges_fwhm, bin_centers_fwhm, d_fwhm_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_FWHM_distribution.root')

    _, _, d_fwhm_inverse_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_FWHM_inverse_distribution.root')
    _, _, d_fwhm_inverse_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_FWHM_inverse_distribution.root')
    bin_edges_fwhm_inverse, bin_centers_fwhm_inverse, d_fwhm_inverse_bibi = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_FWHM_inverse_distribution.root')


    bin_edges_2l = bin_edges[0] / 1e3  # [m]
    bin_centers_2l = bin_centers[0] / 1e3  # [m]
    bin_edges_p_tilde = bin_edges[1]
    bin_centers_p_tilde = bin_centers[1]

    d_2l_p_all = d_2l_p_tbtb + d_2l_p_tbbi + d_2l_p_bibi
    normalization = np.trapezoid(np.trapezoid(d_2l_p_all, x=bin_centers_2l, axis=0), x=bin_centers_p_tilde)
    d_2l_p_all /= normalization
    d_2l_p_tbtb, d_2l_p_tbbi, d_2l_p_bibi = d_2l_p_tbtb / normalization, d_2l_p_tbbi / normalization, d_2l_p_bibi / normalization

    d_2l_tbtb = np.trapezoid(d_2l_p_tbtb, x=bin_centers_p_tilde, axis=1)
    d_2l_tbbi = np.trapezoid(d_2l_p_tbbi, x=bin_centers_p_tilde, axis=1)
    d_2l_bibi = np.trapezoid(d_2l_p_bibi, x=bin_centers_p_tilde, axis=1)

    m_2l_tbtb = np.trapezoid(d_2l_tbtb * bin_centers_2l, x=bin_centers_2l) / np.trapezoid(d_2l_tbtb, x=bin_centers_2l)
    m_2l_tbbi = np.trapezoid(d_2l_tbbi * bin_centers_2l, x=bin_centers_2l) / np.trapezoid(d_2l_tbbi, x=bin_centers_2l)
    m_2l_bibi = np.trapezoid(d_2l_bibi * bin_centers_2l, x=bin_centers_2l) / np.trapezoid(d_2l_bibi, x=bin_centers_2l)

    d_fwhm_all = d_fwhm_tbtb + d_fwhm_tbbi + d_fwhm_bibi
    n_fwhm_all = np.trapezoid(d_fwhm_all, x=bin_centers_fwhm)
    d_fwhm_all, d_fwhm_tbtb, d_fwhm_tbbi, d_fwhm_bibi = d_fwhm_all / n_fwhm_all, d_fwhm_tbtb / n_fwhm_all, d_fwhm_tbbi / n_fwhm_all, d_fwhm_bibi / n_fwhm_all

    d_fwhm_inverse_all = d_fwhm_inverse_tbtb + d_fwhm_inverse_tbbi + d_fwhm_inverse_bibi
    n_fwhm_inverse_all = np.trapezoid(d_fwhm_inverse_all, x=bin_centers_fwhm_inverse)
    d_fwhm_inverse_all, d_fwhm_inverse_tbtb, d_fwhm_inverse_tbbi, d_fwhm_inverse_bibi = d_fwhm_inverse_all / n_fwhm_inverse_all, d_fwhm_inverse_tbtb / n_fwhm_inverse_all, d_fwhm_inverse_tbbi / n_fwhm_inverse_all, d_fwhm_inverse_bibi / n_fwhm_inverse_all

    extent = (bin_edges_2l[0], bin_edges_2l[-1], bin_edges_p_tilde[0], bin_edges_p_tilde[-1])
    aspect_ratio = (bin_edges_2l[-1] - bin_edges_2l[0]) / (bin_edges_p_tilde[-1] - bin_edges_p_tilde[0])

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(12, 4.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 0.05, 1], height_ratios=[1, 3], hspace=0.05)
    axlu = fig.add_subplot(gs[0, 0])
    axll = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    axr = fig.add_subplot(gs[:, 2])

    axlu.stairs(d_2l_tbtb, edges=bin_edges_2l, color='tab:orange', label='TB-TB')
    axlu.stairs(d_2l_tbbi, edges=bin_edges_2l, color='tab:green', label='TB-BI')
    axlu.stairs(d_2l_bibi, edges=bin_edges_2l, color='tab:red', label='BI-BI')
    axlu.set_xlim(bin_edges_2l[0], bin_edges_2l[-1])
    axlu.set_xticklabels([])
    axlu.set_ylabel('PDF')
    # axlu.legend(frameon=False)

    im = axll.imshow(d_2l_p_all.T, origin='lower', extent=extent)
    c_bar = fig.colorbar(im, cax=cax, orientation='vertical')
    c_bar.set_label('PDF')
    axll.set_ylim(-0.83, 0.33)
    axll.set_yticks([-0.5, 0.])
    axll.set_aspect('auto')
    axll.set_xlabel(r'$2L$ [m]')
    axll.set_ylabel(r'$\tilde{p}$')
    axll.text(m_2l_tbtb, 0.2, 'TB-TB', fontweight='bold', ha='center', color='tab:orange')
    axll.text(m_2l_tbbi, -0.8, 'TB-BI', fontweight='bold', ha='center', color='tab:green')
    axll.text(m_2l_bibi, 0.2, 'BI-BI', fontweight='bold', ha='center', color='tab:red')

    axr.stairs(d_fwhm_tbtb, edges=bin_edges_fwhm, color='tab:orange', label='TB-TB')
    axr.stairs(d_fwhm_tbbi, edges=bin_edges_fwhm, color='tab:green', label='TB-BI')
    axr.stairs(d_fwhm_bibi, edges=bin_edges_fwhm, color='tab:red', label='BI-BI')
    axr.set_xlim(0.0, 4)
    axr.set_ylim(0, im.get_clim()[1] / 10)
    axr.set_xlabel(r'$\mathrm{FWHM}_\mathrm{ncl}$ [mm]')
    axr.set_ylabel(r'PDF')
    axr.legend(loc='lower right', frameon=False)

    axr_insert = fig.add_axes([0.75, 0.65, 0.13, 0.2])
    axr_insert.stairs(d_fwhm_inverse_tbtb, edges=bin_edges_fwhm_inverse, color='tab:orange', label='TB-TB')
    axr_insert.stairs(d_fwhm_inverse_tbbi, edges=bin_edges_fwhm_inverse, color='tab:green', label='TB-BI')
    axr_insert.stairs(d_fwhm_inverse_bibi, edges=bin_edges_fwhm_inverse, color='tab:red', label='BI-BI')
    axr_insert.set_xlim(0, 2)
    # axr_insert.set_xticks([0, 2])
    axr_insert.set_yticks([0, 1, 2, 3])
    axr_insert.set_xlabel(r'$\mathrm{FWHM}_\mathrm{ncl}^{-1}$ [mm$^{-1}$]')
    axr_insert.set_ylabel('PDF')

    plt.show()

    return 0


if __name__ == "__main__":
    main()
