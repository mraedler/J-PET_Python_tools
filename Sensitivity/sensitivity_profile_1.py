"""
Sensitivity comparison of true events
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from load_root_hist import load_tb_j_pet_brain_insert_th3d, load_tb_j_pet_brain_insert_thnd


def main():
    # Output directory from the post-processing
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'

    # Simulations with 200 ps CTR
    tbtb_6_30_true, tbbi_6_30_true, bibi_6_30_true, z_edges = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source_200_ps_6_30_mm/2026-02-03_17-22-58_copy/', 'true', return_grid=True)
    tbtb_4_18_true, tbbi_4_18_true, bibi_4_18_true = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'true', return_grid=False)

    # # Simulation with 400 ps CTR
    # tbtb_4_18_true_2, tbbi_4_18_true_2, bibi_4_18_true_2 = load_tb_j_pet_brain_insert_thnd(
    #     output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'true', return_grid=False)

    bibi_factor = np.sum(bibi_4_18_true * bibi_6_30_true) / np.sum(bibi_6_30_true ** 2)
    print(bibi_factor)

    # Get the reference value
    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs((tbtb_6_30_true + tbbi_6_30_true + bibi_6_30_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue', label='ALL')
    ax.stairs(tbtb_6_30_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange', label='TB-TB')
    ax.stairs(tbbi_6_30_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green', label='TB-BI')
    ax.stairs(bibi_6_30_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red', label='BI-BI')

    ax.stairs((tbtb_4_18_true + tbbi_4_18_true + bibi_4_18_true) / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_4_18_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_4_18_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_4_18_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    # # For reference
    # ax.stairs((tbtb_4_18_true_2 + tbbi_4_18_true_2 + bibi_4_18_true_2) / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:blue')
    # ax.stairs(tbtb_4_18_true_2 / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:orange')
    # ax.stairs(tbbi_4_18_true_2 / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:green')
    # ax.stairs(bibi_4_18_true_2 / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:red')

    legend_1 = ax.legend(loc='center', frameon=False, ncol=1)

    h_dummy_1, = plt.plot(np.nan, color='black', linestyle='-')
    h_dummy_2, = plt.plot(np.nan, color='black', linestyle='--')

    ax.legend([h_dummy_1, h_dummy_2], ['Variant 1', 'Variant 2'], frameon=False)
    ax.add_artist(legend_1)

    ax_twin = ax.twinx()
    ax_twin.set_ylim(np.array(ax.get_ylim()) / 10)
    ax_twin.set_ylabel('Sensitivity [%]')

    ax.set_xlabel(r'$z$ [mm]')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
