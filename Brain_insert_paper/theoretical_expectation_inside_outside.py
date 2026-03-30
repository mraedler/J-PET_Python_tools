"""

"""
import sys
import numpy as np
from uproot import open as open_root
import matplotlib.pyplot as plt


def main():
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'
    # root_file = open_root(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBTB_true2.root')

    # bin_edges, bin_centers, d_tbtb = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBTB_t_distribution.root')
    # bin_edges, bin_centers, d_tbbi = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/TBBI_t_distribution.root')
    # bin_edges, bin_centers, d_bibi = load_thnd(output_dir + '/Derenzo_outside_400_ps_6_30_mm/2026-02-17_21-43-59/BIBI_t_distribution.root')
    #
    bin_edges, bin_centers, d_tbtb = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TBTB_t_distribution.root')
    bin_edges, bin_centers, d_tbbi = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/TBBI_t_distribution.root')
    bin_edges, bin_centers, d_bibi = load_thnd(output_dir + '/Derenzo_400_ps_6_30_mm/2026-02-13_10-41-09/BIBI_t_distribution.root')

    d_all = d_tbtb + d_tbbi + d_bibi

    #
    # d = d_all / np.sum(d_all)

    # edges to {-1, 1} mapping
    l_over_ell = 2 * (bin_centers - 1 / 2)

    normalization = np.trapezoid(d_all, x=l_over_ell)
    d_all /= normalization
    d_tbtb, d_tbbi, d_bibi = d_tbtb / normalization, d_tbbi / normalization, d_bibi / normalization

    tau = 6 / 2  # half the crystal pitch

    # Different FWHM approaches
    fwhm_def = tau * (1 + np.abs(l_over_ell))
    fwhm_std = tau * np.sqrt(1 + l_over_ell ** 2)
    fwhm_pct = tau * (2 - np.sqrt(1 - l_over_ell ** 2))
    fwhm_pct[np.abs(l_over_ell) > 3/5] = tau * 3 * (1 + np.abs(l_over_ell)[np.abs(l_over_ell) > 3/5]) / 4

    print(np.trapezoid(d_all * fwhm_std, x=l_over_ell))
    print(np.trapezoid(d_all * fwhm_pct, x=l_over_ell))

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(bin_centers, arr)
    ax.stairs(d_all, 2 * bin_edges - 1, label='ALL')
    ax.stairs(d_tbtb, 2 * bin_edges - 1, label='TB-TB')
    ax.stairs(d_tbbi, 2 * bin_edges - 1, label='TB-BI')
    ax.stairs(d_bibi, 2 * bin_edges - 1, label='BI-BI')
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r'$l/\ell$')
    ax.set_ylabel('PDF')
    legend = ax.legend(loc='upper left', frameon=False)

    ax_twin = ax.twinx()
    ax_twin.plot(l_over_ell, fwhm_def / tau, color='k', linestyle='-', label=r'$\mathrm{FWHM}_\mathrm{def}$')
    ax_twin.plot(l_over_ell, fwhm_std / tau, color='k', linestyle='--', label=r'$\mathrm{FWHM}_\mathrm{std}$')
    ax_twin.plot(l_over_ell, fwhm_pct / tau, color='k', linestyle=':', label=r'$\mathrm{FWHM}_\mathrm{pct}$')
    ax_twin.set_ylim(1, 2.5)
    ax_twin.set_yticks([1, 1.5, 2., 2.5])
    ax_twin.set_ylabel(r'$\mathrm{FWHM}/\tau$')
    ax_twin.legend(loc='upper right', frameon=False)

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
