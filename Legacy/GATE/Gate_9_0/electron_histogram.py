"""
Electron histogram

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from uproot import open
from tqdm import tqdm

# Auxiliary functions
from tree_merger import load_or_get_paths, process_statistics


def main():
    # root_file_name = '/home/martin/J-PET/Gate_mac_9.0/TB_J-PET_Brain_2/Output/2024-02-15_11-27-26/results.root'
    root_file_name = '/home/martin/J-PET/Gate_mac_9.0/TB_J-PET_Brain_2/Output/2024-02-15_11-28-03/results.root'
    root_file = open(root_file_name)

    # Get the hits
    hits = root_file['Hits']

    # Change the positional index to a linear one
    dims = [2, 24, 7, 3570]
    pos_idx_hits = np.ravel_multi_index((np.array(hits['gantryID']), np.array(hits['rsectorID']), np.array(hits['crystalID']), np.array(hits['layerID'])), dims)

    # Build a processID
    process_name = np.array(hits['processName'])
    process_id = process_statistics(process_name, show_histogram=False)

    (entry_indices_paths, pos_idx_paths, pos_x_paths, pos_y_paths, pos_z_paths, time_paths, e_dep_paths, process_id_paths,
     eid_paths, tid_paths, pid_paths, pdg_paths) = load_or_get_paths(hits, pos_idx_hits, process_id, root_file_name)

    electron_pos_x_initial = []
    electron_pos_y_initial = []

    # Find electron initial positions
    for ii in tqdm(range(len(pos_x_paths))):
        if pdg_paths[ii] == 11:
            electron_pos_x_initial.append(pos_x_paths[ii][0])
            electron_pos_y_initial.append(pos_y_paths[ii][0])

    #
    fig, ax = plt.subplots()
    ax.scatter(electron_pos_x_initial, electron_pos_y_initial)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
