"""
Analyze sensitivity maps from CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from uproot import open as open_root
from matplotlib import pyplot as plt

# Auxiliary functions
from read_interfile import read_interfile
from vis import vis_3d
sys.path.append('../')
from sensitivity_coincidences import get_sensitivity, plot_sensitivity
from data_structures import load_or_convert_to_structured_array


def main():
    sens_edit = read_interfile('/home/martin/J-PET/CASToR_RECONS/SENS_MAPS/CONST.hdr')

    print(sens_edit)

    sys.exit()






    _, _, _, sens = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/subdivided/TB_brain_2_tbb_GATE.hdr', return_grid=True)
    vis_3d(sens, axis=1)
    sys.exit()



    # Binning for the line sensitivity
    n_bins = 80
    z_edges = np.linspace(-1200., 1200., n_bins + 1)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_widths = z_edges[1:] - z_edges[:-1]

    # Normalization
    activity = 1000.  # [kBq]
    run_time = 100.  # [s]
    source_length = 2400  # [mm]
    normalization = activity / (source_length / z_widths[0]) * run_time


    # Line source sensitivity
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-05-28_15-27-04/results.root')  # TB-B: 100 s
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-06-09_12-04-35/results.root')  # TB-B_2: 100 s
    root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Frontal_detector/2024-06-24_17-30-58/results.root')  # Frontal det.: 100 s
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'])
    # coincidences_struct = equate_layer_utilization(coincidences_struct)
    h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain = get_sensitivity(coincidences_struct, z_edges)
    # plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain, normalization, [-815 - 330 / 2, -815 + 330 / 2])
    plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain, normalization, [-1247 - 64 / 2, -1247 + 64 / 2])
    sys.exit()


    # Load the sensitivity maps
    _, _, z, sens = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_only_CASToR.hdr', return_grid=True)
    sens_edit = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_brain_CONST.hdr')
    sens_brain = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_brain_CASToR.hdr')
    _, _, z2, sens_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_brain_GATE.hdr', return_grid=True)

    # Plot central profiles
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.plot(z, sens[10, 10, :], label='TB only CASToR')
    ax.plot(z, sens_brain[10, 10, :], label='TB-brain CASToR')
    ax.plot(z2, sens_gate[50, 50, :] * 4e2, label='TB-brain GATE')
    # ax.plot(z2, sens_edit[10, 10, :] * 1e6, label='Constant')
    ax.set_xlabel(r'$z$ [mm]')
    ax.legend(loc='lower center', ncol=1)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
