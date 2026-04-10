"""

"""
import sys
import numpy as np
from uproot import open as open_root
import matplotlib.pyplot as plt


def main():
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'
    # root_file = open_root(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBTB_true2.root')

    # bin_edges, bin_centers, d_tbtb = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBTB_p_distribution.root')
    # bin_edges, bin_centers, d_tbbi = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBBI_p_distribution.root')
    # bin_edges, bin_centers, d_bibi_6_30 = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/BIBI_p_distribution.root')

    bin_edges, bin_centers, d_tbtb_6_30 = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TBTB_p_distribution.root')
    bin_edges, bin_centers, d_tbbi_6_30 = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TBBI_p_distribution.root')
    bin_edges, bin_centers, d_bibi_6_30 = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/BIBI_p_distribution.root')

    bin_edges, bin_centers, d_tbtb_4_18 = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBTB_p_distribution.root')
    bin_edges, bin_centers, d_tbbi_4_18 = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TBBI_p_distribution.root')
    bin_edges, bin_centers, d_bibi_4_18 = load_thnd(output_dir + '/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/BIBI_p_distribution.root')

    d_all_6_30 = d_tbtb_6_30 + d_tbbi_6_30 + d_bibi_6_30
    d_all_4_18 = d_tbtb_4_18 + d_tbbi_4_18 + d_bibi_4_18

    # edges to {-1, 1} mapping
    p_over_l = 2 * (bin_centers - 1 / 2)

    norm_6_30 = np.trapezoid(d_all_6_30, x=p_over_l)
    d_all_6_30 /= norm_6_30
    d_tbtb_6_30, d_tbbi_6_30, d_bibi_6_30 = d_tbtb_6_30 / norm_6_30, d_tbbi_6_30 / norm_6_30, d_bibi_6_30 / norm_6_30

    norm_4_18 = np.trapezoid(d_all_4_18, x=p_over_l)
    d_all_4_18 /= norm_4_18
    d_tbtb_4_18, d_tbbi_4_18, d_bibi_4_18 = d_tbtb_4_18 / norm_4_18, d_tbbi_4_18 / norm_4_18, d_bibi_4_18 / norm_4_18

    # # Different FWHM approaches
    # d = 6 / 2  # half the crystal size
    # fwhm_def = d * (1 + np.abs(p_over_l))
    # fwhm_std = d * np.sqrt(1 + p_over_l ** 2)
    # fwhm_pct = d * (2 - np.sqrt(1 - p_over_l ** 2))
    # fwhm_pct[np.abs(p_over_l) > 3/5] = d * 3 * (1 + np.abs(p_over_l)[np.abs(p_over_l) > 3/5]) / 4

    d_m1 = 2
    d_p1 = 3
    fwhm_1 = d_m1 * np.sqrt(1 + p_over_l ** 2)
    fwhm_2 = d_p1 * np.sqrt(1 + p_over_l ** 2)
    fwhm_12 = np.sqrt((1 - p_over_l) ** 2 * d_m1 ** 2 / 2 + (1 + p_over_l) ** 2 * d_p1 ** 2 / 2)

    n_all_6_30 = np.trapezoid(d_all_6_30, x=p_over_l)
    n_tbtb_6_30 = np.trapezoid(d_tbtb_6_30, x=p_over_l)
    n_tbbi_6_30 = np.trapezoid(d_tbbi_6_30, x=p_over_l)
    n_bibi_6_30 = np.trapezoid(d_bibi_6_30, x=p_over_l)

    fwhm_tbtb_6_30 = np.trapezoid(d_tbtb_6_30 * fwhm_2, x=p_over_l) / n_tbtb_6_30
    fwhm_tbbi_6_30 = np.trapezoid(d_tbbi_6_30 * fwhm_2, x=p_over_l) / n_tbbi_6_30
    fwhm_bibi_6_30 = np.trapezoid(d_bibi_6_30 * fwhm_2, x=p_over_l) / n_bibi_6_30

    print('Variant 1\n---------')
    # print('ALL a: %1.2f mm (%1.1f %%)' % (np.trapezoid(d_all_6_30 * fwhm_2, x=p_over_l) / n_all_6_30, n_all_6_30 * 100))
    print('ALL a: %1.2f mm (%1.1f %%)' % ((n_tbtb_6_30 * fwhm_tbtb_6_30 + n_tbbi_6_30 * fwhm_tbbi_6_30 + n_bibi_6_30 * fwhm_bibi_6_30) / n_all_6_30, n_all_6_30 * 100))
    print('ALL h: %1.2f mm (%1.1f %%)' % (n_all_6_30 / (n_tbtb_6_30 / fwhm_tbtb_6_30 + n_tbbi_6_30 / fwhm_tbbi_6_30 + n_bibi_6_30 / fwhm_bibi_6_30), n_all_6_30 * 100))
    print('TB-TB: %1.2f mm (%1.1f %%)' % (fwhm_tbtb_6_30, n_tbtb_6_30 * 100))
    print('TB-BI: %1.2f mm (%1.1f %%)' % (fwhm_tbbi_6_30, n_tbbi_6_30 * 100))
    print('BI-BI: %1.2f mm (%1.1f %%)\n' % (fwhm_bibi_6_30, n_bibi_6_30 * 100))

    n_all_4_18 = np.trapezoid(d_all_4_18, x=p_over_l)
    n_tbtb_4_18 = np.trapezoid(d_tbtb_4_18, x=p_over_l)
    n_tbbi_4_18 = np.trapezoid(d_tbbi_4_18, x=p_over_l)
    n_bibi_4_18 = np.trapezoid(d_bibi_4_18, x=p_over_l)

    fwhm_tbtb_4_18 = np.trapezoid(d_tbtb_4_18 * fwhm_2, x=p_over_l) / n_tbtb_4_18
    fwhm_tbbi_4_18 = np.trapezoid(d_tbbi_4_18 * fwhm_12, x=p_over_l) / n_tbbi_4_18
    fwhm_bibi_4_18 = np.trapezoid(d_bibi_4_18 * fwhm_1, x=p_over_l) / n_bibi_4_18

    print('Variant 2\n---------')
    print('ALL a: %1.2f mm (%1.1f %%)' % ((n_tbtb_4_18 * fwhm_tbtb_4_18 + n_tbbi_4_18 * fwhm_tbbi_4_18 + n_bibi_4_18 * fwhm_bibi_4_18) / n_all_4_18, n_all_4_18 * 100))
    print('ALL h: %1.2f mm (%1.1f %%)' % (n_all_4_18 / (n_tbtb_4_18 / fwhm_tbtb_4_18 + n_tbbi_4_18 / fwhm_tbbi_4_18 + n_bibi_4_18 / fwhm_bibi_4_18), n_all_4_18 * 100))
    print('TB-TB: %1.2f mm (%1.1f %%)' % (fwhm_tbtb_4_18, n_tbtb_4_18 * 100))
    print('TB-BI: %1.2f mm (%1.1f %%)' % (fwhm_tbbi_4_18, n_tbbi_4_18 * 100))
    print('BI-BI: %1.2f mm (%1.1f %%)\n' % (fwhm_bibi_4_18, n_bibi_4_18 * 100))

    print('Mean of TB-BI\n-------------')
    print('Variant 1: %1.3f' % (np.trapezoid(d_tbbi_6_30 * p_over_l, x=p_over_l) / n_tbbi_6_30))
    print('Variant 2: %1.3f' % (np.trapezoid(d_tbbi_4_18 * p_over_l, x=p_over_l) / n_tbbi_4_18))

    sys.exit()

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8))
    ax0.stairs(d_all_6_30, 2 * bin_edges - 1, color='tab:blue', label='ALL')
    ax0.stairs(d_tbtb_6_30, 2 * bin_edges - 1, color='tab:orange', label='TB-TB')
    ax0.stairs(d_tbbi_6_30, 2 * bin_edges - 1, color='tab:green', label='TB-BI')
    ax0.stairs(d_bibi_6_30, 2 * bin_edges - 1, color='tab:red', label='BI-BI')
    ax0.set_xlim(-1, 1)
    ax0.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax0.set_xticklabels([])
    ax0.set_ylim(0, 3)
    ax0.set_yticks([0, 1, 2, 3])
    ax0.set_ylabel('PDF')
    ax0.legend(loc='upper left', frameon=True, ncol=4)
    ax0.text(-0.96, 2.7, '(a) Variant 1', ha='left')
    # ax0.text(-1.15, -0.1, '(a)', ha='right', va='top')

    ax0_twin = ax0.twinx()
    # ax0_twin.plot(p_over_l, fwhm_1, color='k', linestyle='-', label=r'$\mathrm{FWHM}(\tilde{p},D_{1})$')
    ax0_twin.plot(p_over_l, fwhm_2, color='k', linestyle='-', label=r'$\mathrm{FWHM}(\tilde{p},D)$')
    # ax0_twin.plot(p_over_l, fwhm_12, color='k', linestyle=':', label=r'$\mathrm{FWHM}(\tilde{p},D_\text{-1},D_1)$')
    ax0_twin.set_ylim(2, 5)
    ax0_twin.set_yticks([2., 3., 4., 5.])
    ax0_twin.set_ylabel(r'$\mathrm{FWHM}$ [mm]')
    ax0_twin.legend(loc='upper right', frameon=False)

    ax1.stairs(d_all_4_18, 2 * bin_edges - 1, color='tab:blue', label='ALL')
    ax1.stairs(d_tbtb_4_18, 2 * bin_edges - 1, color='tab:orange', label='TB-TB')
    ax1.stairs(d_tbbi_4_18, 2 * bin_edges - 1, color='tab:green', label='TB-BI')
    ax1.stairs(d_bibi_4_18, 2 * bin_edges - 1, color='tab:red', label='BI-BI')
    ax1.set_xlim(-1, 1)
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_ylim(0, 3)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_xlabel(r'$\tilde{p}=p/L$')
    ax1.set_ylabel('PDF')
    ax1.text(-0.96, 2.7, '(b) Variant 2', ha='left')
    # ax1.text(-1.15, -0.1, '(b)', ha='right', va='top')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(p_over_l, fwhm_2, color='k', linestyle='-', label=r'$\mathrm{FWHM}(\tilde{p},D_1)$')
    ax1_twin.plot(p_over_l, fwhm_12, color='k', linestyle='--', label=r'$\mathrm{FWHM}(\tilde{p},D_\text{-1},D_1)$')
    ax1_twin.plot(p_over_l, fwhm_1, color='k', linestyle=':', label=r'$\mathrm{FWHM}(\tilde{p},D_\text{-1})$')
    ax1_twin.set_ylim(2, 5)
    ax1_twin.set_yticks([2., 3., 4., 5.])
    ax1_twin.set_ylabel(r'$\mathrm{FWHM}$ [mm]')
    ax1_twin.legend(loc='upper right', frameon=False)
    plt.show()

    return 0


def load_thnd(root_file_path):
    root_file = open_root(root_file_path)
    data = root_file['THnD'].tojson()

    #
    # print(data.keys())
    n_dim = data['fNdimensions']
    n_entries = data['fEntries']

    # Axes
    n_bins = np.array([data['fAxes']['arr'][d]['fNbins'] for d in range(n_dim)])
    x_min = np.array([data['fAxes']['arr'][d]['fXmin'] for d in range(n_dim)])
    x_max = np.array([data['fAxes']['arr'][d]['fXmax'] for d in range(n_dim)])

    bin_edges = [np.linspace(x_min[ii], x_max[ii], n_bins[ii] + 1) for ii in range(n_bins.size)]
    bin_centers = [(bin_edges[ii][:-1] + bin_edges[ii][1:]) / 2 for ii in range(n_bins.size)]

    # Array
    arr = np.array(data['fArray']['fData']).reshape(n_bins + 2)

    # Alternatively load with the stride
    # print(data['fArray']['_typename'])
    # arr = np.array(data["fArray"]['fData'])
    # index_strides = np.array(data['fArray']['fSizes'])
    # byte_strides = tuple(index_strides[1:])
    # arr = np.lib.stride_tricks.as_strided(arr, shape=n_bins + 2, strides=byte_strides)

    # Remove the extra dimensions that collect entries outside the histogram
    # print((arr[0] + arr[-1]) / np.sum(arr[1:-1]))
    arr = arr[(slice(1, -1),) * n_dim]
    # arr = arr.squeeze()

    # Remove the list if arr is 1D
    if len(arr.shape) == 1:
        bin_edges = bin_edges[0]
        bin_centers = bin_centers[0]

    return bin_edges, bin_centers, arr


if __name__ == '__main__':
    main()
