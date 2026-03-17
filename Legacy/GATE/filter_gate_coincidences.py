"""
Filter GATE coincidences

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import split
import matplotlib.pyplot as plt
import numpy as np
from uproot import open as open_root
from tqdm import tqdm

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array


def main():
    root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-04-11_10-03-38/results.root'
    root_file = open_root(root_path)
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'])
    # np.save(split(root_path)[0] + '/merged_coincidences.npy', coincidences_struct)
    # print(coincidences_struct.dtype)

    # Determine the true coincidences
    true_coincidences = ((coincidences_struct['eventID1'] == coincidences_struct['eventID2'])
                         & (coincidences_struct['comptonCrystal1'] == 1)
                         & (coincidences_struct['comptonCrystal2'] == 1))

    # Plot
    source_pos_z1 = coincidences_struct['sourcePosZ1']
    source_pos_z2 = coincidences_struct['sourcePosZ2']

    z_edges = np.arange(-1215, 1215 + 1, 15)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_width = z_edges[1:] - z_edges[:-1]

    h1_all, _ = np.histogram(source_pos_z1, bins=z_edges)
    h2_all, _ = np.histogram(source_pos_z2, bins=z_edges)
    h_all = h1_all + h2_all

    h1_true, _ = np.histogram(source_pos_z1[true_coincidences], bins=z_edges)
    h2_true, _ = np.histogram(source_pos_z2[true_coincidences], bins=z_edges)
    h_true = h1_true + h2_true

    percentage_true = np.zeros(h_all.shape)
    percentage_true[h_all > 0] = h_true[h_all > 0] / h_all[h_all > 0]

    percentage_true_av = np.sum(true_coincidences) / true_coincidences.size

    fig, ax = plt.subplots()
    ax.bar(z_centers, h_all, width=z_width)
    ax.bar(z_centers, h_true, width=z_width)
    ax_twin = ax.twinx()
    ax_twin.plot(z_centers, percentage_true, color='black')
    ax_twin.plot(z_centers, percentage_true_av * np.ones(z_centers.shape), color='black', linestyle='--')
    ax_twin.set_ylim(0.75, 0.85)
    plt.show()

    # np.save(split(root_path)[0] + '/merged_coincidences_true.npy', coincidences_struct[true_coincidences])

    return 0


if __name__ == "__main__":
    main()
