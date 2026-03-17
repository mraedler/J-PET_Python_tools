"""
Sensitivity plots

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from uproot import open
from matplotlib import pyplot as plt


def sensitivity_plot():
    # Histogram binning
    # n_bins = 61
    # z_edges = np.linspace(-915., 915., n_bins + 1)
    n_bins = 80
    z_edges = np.linspace(-1200., 1200., n_bins + 1)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_widths = z_edges[1:] - z_edges[:-1]

    # Normalization
    activity = 1000.  # kBq
    run_time = 10.  # s
    normalization = activity / n_bins * run_time

    # # root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_11-49-02/results.root'  # before bug fix
    # # root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_11-54-44/results.root'  # after bug fix
    # plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain, normalization, vertical_lines=[-815. - 330. / 2, -815. + 330. / 2])


    # root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-03-12_14-33-57/results.root'  # frontal
    # root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-03-12_15-12-50/results.root'  # insert
    root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-04-11_10-03-38/results.root'  # TB only
    h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain = get_sensitivity(root_path, z_edges)
    # plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate,
    #                  h_filtered_brain, normalization, vertical_lines=[-1247. - 64. / 2, -1247. + 64. / 2])
    plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate,
                     h_filtered_brain, normalization, vertical_lines=[-815. - 330. / 2, -815. + 330. / 2])


    sys.exit()

    # Compare different SiPM sizes
    h_raw_6, h_filtered_6, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_12-51-15/results.root', z_edges)
    h_raw_4, h_filtered_4, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_17-56-13/results.root', z_edges)
    h_raw_3, h_filtered_3, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_19-15-01/results.root', z_edges)

    compare_sensitivities(z_edges,
                          [h_raw_6, h_raw_4, h_raw_3],
                          [h_filtered_6, h_filtered_4, h_filtered_3],
                          [r'$6\times6$ mm', r'$4\times4$ mm', r'$3\times3$ mm'], normalization)

    # Compare different scintillator thicknesses
    h_raw_3, h_filtered_3, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_12-51-15/results.root', z_edges)
    h_raw_2, h_filtered_2, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-23_14-53-21/results.root', z_edges)
    h_raw_1, h_filtered_1, _, _, _ = get_sensitivity('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-23_16-06-58/results.root', z_edges)

    compare_sensitivities(z_edges,
                          [h_raw_3, h_raw_2, h_raw_1],
                          [h_filtered_3, h_filtered_2, h_filtered_1],
                          ['3 cm', '2 cm', '1 cm'], normalization)

    return 0


def get_sensitivity(input_var, z_edges):
    if isinstance(input_var, str):
        # Get the coincidences
        root_file = open(input_var)
        # print(root_file.keys())
        coincidences = root_file['MergedCoincidences']
    else:
        coincidences = input_var  # todo: fixme

    # Source position (could be either 1 or 2)
    source_pos_z1 = np.array(coincidences['sourcePosZ1'])
    # source_pos_z2 = np.array(coincidences['sourcePosZ2'])
    data_raw = source_pos_z1.copy()
    # data_raw = source_pos_z2.copy()

    # Characteristics for filtering
    compton_crystal1 = np.array(coincidences['comptonCrystal1'])
    compton_crystal2 = np.array(coincidences['comptonCrystal2'])
    gantry_id1 = np.array(coincidences['gantryID1'])
    gantry_id2 = np.array(coincidences['gantryID2'])

    # Logical indexing for filtering
    both_first_compton = (compton_crystal1 == 1) & (compton_crystal2 == 1)
    both_in_total_body_scanner = (gantry_id1 == 0) & (gantry_id2 == 0)
    # both_in_separate_scanners = ((gantry_id1 == 0) & (gantry_id2 == 1)) | ((gantry_id1 == 1) & (gantry_id2 == 0))
    both_in_separate_scanners = gantry_id1 != gantry_id2
    both_in_brain_scanner = (gantry_id1 == 1) & (gantry_id2 == 1)

    # Calculate the histograms
    h_raw, _ = np.histogram(data_raw, bins=z_edges)
    h_filtered, _ = np.histogram(data_raw[both_first_compton], bins=z_edges)
    h_filtered_total_body, _ = np.histogram(data_raw[both_first_compton & both_in_total_body_scanner], bins=z_edges)
    h_filtered_separate, _ = np.histogram(data_raw[both_first_compton & both_in_separate_scanners], bins=z_edges)
    h_filtered_brain, _ = np.histogram(data_raw[both_first_compton & both_in_brain_scanner], bins=z_edges)

    return h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain


def plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain, normalization, vertical_lines=[]):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(z_centers, h_filtered / normalization, width=z_widths, label='TOT')
    ax.bar(z_centers, h_filtered_total_body / normalization, width=z_widths, label='TB-TB')
    ax.bar(z_centers, h_filtered_separate / normalization, width=z_widths, label='TB-B')
    ax.bar(z_centers, h_filtered_brain / normalization, width=z_widths, label='B-B')

    ax.stairs(h_filtered_total_body / normalization, edges=z_edges, color='tab:orange')
    ax.stairs(np.flip(h_filtered_total_body / normalization), edges=z_edges, linestyle='--', color='tab:orange')

    s_raw = ax.stairs(h_raw / normalization, edges=z_edges, linestyle='--', color='black')
    s_compton = ax.stairs(h_filtered / normalization, edges=z_edges, color='black')

    # y_lim = [0, np.ceil(np.max(h_raw / normalization) / 10) * 10]
    y_lim = [0, 65]
    for x_vertical in vertical_lines:
        ax.plot([x_vertical, x_vertical], y_lim, linestyle=':', color='black')
    ax.set_xlim(z_edges[0] * 1.1, z_edges[-1] * 1.1)
    ax.set_ylim(y_lim)

    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel('Sensitivity [cps/kBq]')
    legend_1 = ax.legend()

    legend_2 = ax.legend([s_raw, s_compton], ['All coincidences', 'True coincidences'], frameon=False, loc='upper center')
    ax.add_artist(legend_1)
    plt.show()
    return 0


def compare_sensitivities(z_edges, h_raw_list, h_filtered_list, labels_list, normalization):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for ii in range(len(h_raw_list)):
        ax.stairs(h_raw_list[ii] / normalization, edges=z_edges, color=color_cycle[ii], label=labels_list[ii])
        ax.stairs(h_filtered_list[ii] / normalization, edges=z_edges, linestyle='--', color=color_cycle[ii])

    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel('Sensitivity [cps/kBq]')
    legend_1 = ax.legend()

    # Dummy plots for the second legend
    solid, = ax.plot(np.nan, color='black')
    dashed, = ax.plot(np.nan, linestyle='--', color='black')
    ax.legend([solid, dashed], ['Raw', 'comptonCrystal == 1'], frameon=False, loc='lower center')
    ax.add_artist(legend_1)

    # ax_twin = ax.twinx()
    # ax_twin.stairs(h_raw_list[1] / h_raw_list[0], edges=z_edges, color='black')
    # ax_twin.stairs(h_raw_list[2] / h_raw_list[1], edges=z_edges, linestyle='--', color='black')
    # ax_twin.stairs(np.ones(z_edges.size - 1), edges=z_edges, linestyle=':', color='black')
    # ax_twin.stairs(np.ones(z_edges.size - 1) * 0.96, edges=z_edges, linestyle=':', color='black')

    plt.show()
    return 0


if __name__ == '__main__':
    sensitivity_plot()
