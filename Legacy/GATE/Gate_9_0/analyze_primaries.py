"""
Analyze the events without secondaries

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.constants import electron_mass, speed_of_light, electron_volt, hbar, fine_structure
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    npz_file = np.load(sys.path[0] + '/data/paths_1MBq_10s_1_primaries_only.npz', allow_pickle=True)
    # entry_indices_paths = npz_file['entry_indices_paths']
    pos_idx_paths = npz_file['pos_idx_paths']
    # pos_x_paths = npz_file['pos_x_paths']
    # pos_y_paths = npz_file['pos_y_paths']
    # pos_z_paths = npz_file['pos_z_paths']
    # time_paths = npz_file['time_paths']
    e_dep_paths = npz_file['e_dep_paths']
    process_id_paths = npz_file['process_id_paths']
    eid_paths = npz_file['event_id_paths']
    tid_paths = npz_file['track_id_paths']
    # pid_paths = npz_file['parent_track_id_paths']
    # pdg_paths = npz_file['pdg_encoding_paths']

    # h = np.bincount(np.concatenate(process_id_paths))
    # fig, ax = plt.subplots()
    # ax.bar(np.arange(h.size), h, width=0.8)
    # plt.show()

    # Initial (and final) indices with shared eventID
    eid_idx = np.argwhere(np.diff(eid_paths) > 0)[:, 0]
    eid_idx = np.insert(eid_idx + 1, 0, 0)
    eid_idx = np.append(eid_idx, eid_paths.size)
    entry_indices = np.arange(eid_paths.size)

    # Allocate
    step_count = []
    compton_count = []
    rayleigh_count = []
    photoelectric_count = []
    energies_initial_compton = []
    energies_deposited_compton = []
    energies_rayleigh = []
    energies_photoelectric = []

    # Loop over events
    for ii in tqdm(range(eid_idx.size - 1)):
        entry_indices_temp = entry_indices[eid_idx[ii]:eid_idx[ii + 1]]
        tid_temp = tid_paths[entry_indices_temp]

        # Loop over tracks
        for jj in entry_indices_temp:
            #
            process_id_paths_temp = process_id_paths[jj]
            e_dep_paths_temp = e_dep_paths[jj]

            # Construct an array of the initial energies of each step
            initial_energy_temp = .511 - np.insert(np.cumsum(e_dep_paths_temp), 0, 0.)
            initial_energy_temp = initial_energy_temp[:-1]

            # Readout
            step_count.append(process_id_paths_temp.size)
            compton_count.append(np.sum(process_id_paths_temp == 2))
            rayleigh_count.append(np.sum(process_id_paths_temp == 0))
            photoelectric_count.append(np.sum(process_id_paths_temp == 6))

            energies_initial_compton.append(initial_energy_temp[process_id_paths_temp == 2])
            energies_deposited_compton.append(e_dep_paths_temp[process_id_paths_temp == 2])
            energies_rayleigh.append(initial_energy_temp[process_id_paths_temp == 0])
            energies_photoelectric.append(initial_energy_temp[process_id_paths_temp == 6])

    # Concatenate
    step_count = np.array(step_count)
    compton_count = np.array(compton_count)
    rayleigh_count = np.array(rayleigh_count)
    photoelectric_count = np.array(photoelectric_count)

    energies_initial_compton = np.concatenate(energies_initial_compton)
    energies_deposited_compton = np.concatenate(energies_deposited_compton)
    energies_rayleigh = np.concatenate(energies_rayleigh)
    energies_photoelectric = np.concatenate(energies_photoelectric)

    np.savez(sys.path[0] + '/data/metadata_primaries.npz',
             step_count=step_count,
             compton_count=compton_count,
             rayleigh_count=rayleigh_count,
             photoelectric_count=photoelectric_count,
             energies_initial_compton=energies_initial_compton,
             energies_deposited_compton=energies_deposited_compton,
             energies_rayleigh=energies_rayleigh,
             energies_photoelectric=energies_photoelectric)

    return 0


if __name__ == '__main__':
    main()
