"""
Plot the one dimensional sensitivity profile

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from load_root_hist import load_tb_j_pet_brain_insert_th3d


def main():
    # Output directory from the post-processing
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'

    # Older version (probably simulated with iterative time-based event selection)
    tbtb_true, tbbi_true, bibi_true, z_edges = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'true', return_grid=True)

    tbtb_true_sharareh, tbbi_true_sharareh, bibi_true_sharareh = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'true_sharareh')

    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs((tbtb_true + tbbi_true + bibi_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue', label='ALL')
    ax.stairs(tbtb_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange', label='TB-TB')
    ax.stairs(tbbi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green', label='TB-BI')
    ax.stairs(bibi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red', label='BI-BI')

    ax.stairs(tbtb_true_sharareh / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange', label='TB-TB')

    plt.show()

    return 0


if __name__ == "__main__":
    main()
