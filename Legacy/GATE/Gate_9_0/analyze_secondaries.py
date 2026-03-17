"""
Analyze the secondaries

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Auxiliary functions
from .analyze_processes import energy_histogram


def main():
    npz_file = np.load(sys.path[0] + '/data/paths_1MBq_10s_1_with_secondaries.npz', allow_pickle=True)
    entry_indices_paths = npz_file['entry_indices_paths']
    pos_idx_paths = npz_file['pos_idx_paths']
    pos_x_paths = npz_file['pos_x_paths']
    pos_y_paths = npz_file['pos_y_paths']
    pos_z_paths = npz_file['pos_z_paths']
    time_paths = npz_file['time_paths']
    e_dep_paths = npz_file['e_dep_paths']
    process_id_paths = npz_file['process_id_paths']
    eid_paths = npz_file['event_id_paths']
    tid_paths = npz_file['track_id_paths']
    pid_paths = npz_file['parent_track_id_paths']
    pdg_paths = npz_file['pdg_encoding_paths']

    # 52702

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
    photons_step_count = []
    compton_count = []
    rayleigh_count = []
    photoelectric_count = []
    is_parent = []
    energies_initial_compton = []
    energies_deposited_compton = []
    energies_rayleigh = []
    energies_photoelectric = []

    electrons_step_count = []
    transportation_count = []
    bremsstrahlung_count = []
    ionization_count = []
    msc_count = []
    electron_energy_deposition = []
    electron_travel_distance = []

    electron_parents = []

    distance_to_closest_interaction = []
    process_of_closest_interaction = []

    orphan_electron_counter = 0
    tertiary_electron_counter = 0
    tertiary_photon_counter = 0

    electron_energy_from_compton = []
    electron_energy_from_photoelectric = []
    electron_path_x_compton, electron_path_y_compton, electron_path_z_compton = [], [], []
    electron_path_x_photoelectric, electron_path_y_photoelectric, electron_path_z_photoelectric = [], [], []

    x_initial_list, y_initial_list, z_initial_list, idx_initial_list = [], [], [], []

    # Loop over events
    orphan_count = 0
    for ii in tqdm(range(eid_idx.size - 1)):
        entry_indices_temp = entry_indices[eid_idx[ii]:eid_idx[ii + 1]]
        pid_temp = pid_paths[entry_indices_temp]
        tid_temp = tid_paths[entry_indices_temp]

        # Loop tracks
        for jj in entry_indices_temp:
            process_id_paths_temp = process_id_paths[jj]
            e_dep_paths_temp = e_dep_paths[jj]

            # Primary photons
            if (pdg_paths[jj] == 22) & (tid_paths[jj] <= 2):
                # Construct an array of the initial energies of each step
                initial_energy_temp = .511 - np.insert(np.cumsum(e_dep_paths_temp), 0, 0.)
                initial_energy_temp = initial_energy_temp[:-1]

                # Readout
                photons_step_count.append(process_id_paths_temp.size)
                compton_count.append(np.sum(process_id_paths_temp == 2))
                rayleigh_count.append(np.sum(process_id_paths_temp == 0))
                photoelectric_count.append(np.sum(process_id_paths_temp == 6))

                energies_initial_compton.append(initial_energy_temp[process_id_paths_temp == 2])
                energies_deposited_compton.append(e_dep_paths_temp[process_id_paths_temp == 2])
                energies_rayleigh.append(initial_energy_temp[process_id_paths_temp == 0])
                energies_photoelectric.append(initial_energy_temp[process_id_paths_temp == 6])

                if tid_paths[jj] in pid_temp:
                    is_parent.append(True)
                else:
                    is_parent.append(False)

            # Secondary electrons
            if pdg_paths[jj] == 11:
                # Readout
                electrons_step_count.append(process_id_paths_temp.size)
                transportation_count.append(np.sum(process_id_paths_temp == 1))
                bremsstrahlung_count.append(np.sum(process_id_paths_temp == 3))
                ionization_count.append(np.sum(process_id_paths_temp == 4))
                msc_count.append(np.sum(process_id_paths_temp == 5))

                # Electron energy deposition
                electron_energy_deposition.append(np.sum(e_dep_paths_temp))

                #
                x_electron, y_electron, z_electron = pos_x_paths[jj], pos_y_paths[jj], pos_z_paths[jj]
                delta = np.sqrt(np.diff(x_electron) ** 2 + np.diff(y_electron) ** 2 + np.diff(z_electron) ** 2)
                electron_travel_distance.append(np.sum(delta))

                # Get the coordinate of the initial electron hit

                x_initial, y_initial, z_initial = x_electron[0], y_electron[0], z_electron[0]
                # if len(x_electron) > 10:
                #     x_initial, y_initial, z_initial = x_electron[10], y_electron[10], z_electron[10]

                # If the parent particle is among the tracks of the current event
                if (pid_paths[jj] in tid_temp) & (pid_paths[jj] <= 2):
                    x_initial_list.append(x_initial)
                    y_initial_list.append(y_initial)
                    z_initial_list.append(z_initial)
                    idx_initial_list.append(pos_idx_paths[jj][0])

                    jj_parent = entry_indices_temp[pid_paths[jj] == tid_temp]
                    pdg_parent = pdg_paths[jj_parent][0]
                    x_parent, y_parent, z_parent = pos_x_paths[jj_parent][0], pos_y_paths[jj_parent][0], pos_z_paths[jj_parent][0]
                    process_id_parent = process_id_paths[jj_parent][0]
                    e_dep_parent = e_dep_paths[jj_parent][0]

                    distances = np.sqrt((x_parent - x_initial) ** 2 + (y_parent - y_initial) ** 2 + (z_parent - z_initial) ** 2)
                    idx_min = np.argmin(distances)
                    distance_to_closest_interaction.append(distances[idx_min])
                    process_of_closest_interaction.append(process_id_parent[idx_min])
                    electron_parents.append(pdg_parent)

                    if distances[idx_min] < 0.2:
                        # print(idx_min)
                        # print(process_id_parent)
                        # print(e_dep_parent)
                        # print()

                        if process_id_parent[idx_min] == 2:
                            electron_energy_from_compton.append(np.sum(e_dep_paths_temp))
                            electron_path_x_compton.append(x_electron)
                            electron_path_y_compton.append(y_electron)
                        elif process_id_parent[idx_min] == 6:
                            electron_energy_from_photoelectric.append(np.sum(e_dep_paths_temp))
                            electron_path_x_photoelectric.append(x_electron)
                            electron_path_y_photoelectric.append(y_electron)


                # If parent is not scored but one of the primary photons
                elif pid_paths[jj] <= 2:
                    orphan_electron_counter += 1

                # Electrons created from secondaries
                elif pid_paths[jj] > 2:
                    tertiary_electron_counter += 1
                    # if pid_paths[jj] > 3:
                    #     print(pid_paths[jj])

            # Secondary photons
            if (pdg_paths[jj] == 22) & (tid_paths[jj] > 2):
                tertiary_photon_counter += 1
                # if pid_paths[jj] > 3:
                    # print(pid_paths[jj])
                # print(tid_paths[jj])
                # print(process_id_paths_temp)
                # print(pid_paths[jj])
                # print(process_id_paths[entry_indices_temp][pid_paths[jj] == tid_temp])
                # print(process_id_paths[entry_indices_temp][pid_paths[jj] == tid_temp])
                pass

            # print(tid_paths[jj])

            #
            x_initial = pos_x_paths[jj][0]
            y_initial = pos_y_paths[jj][0]
            z_initial = pos_z_paths[jj][0]
            time_initial = time_paths[jj][0]
            pdg_initial = pdg_paths[jj]

            # print(x_initial, y_initial, z_initial, time_initial, pdg_initial)
            # print()

            # # Check if the parent of the current secondary is among the hits scored in this event
            # if (pid_paths[jj] in tid_temp) and pdg_initial == 11:
            #     idx = entry_indices_temp[tid_temp == pid_paths[jj]]
            #     if np.any(process_id_paths[idx][0] == 6):
            #         print(x_initial, y_initial, z_initial, time_initial, pdg_initial)
            #         print()
            #
            #         print(pos_x_paths[idx])
            #         print(pos_y_paths[idx])
            #         print(pos_z_paths[idx])
            #         print(e_dep_paths[idx])
            #         print(process_id_paths[idx])
            #         print()
            # else:
            #     # print(np.sum(e_dep_paths[jj]))
            #     orphan_count += 1

    # Concatenate
    photons_step_count = np.array(photons_step_count)
    compton_count = np.array(compton_count)
    rayleigh_count = np.array(rayleigh_count)
    photoelectric_count = np.array(photoelectric_count)
    is_parent = np.array(is_parent)

    energies_initial_compton = np.concatenate(energies_initial_compton)
    energies_deposited_compton = np.concatenate(energies_deposited_compton)
    energies_rayleigh = np.concatenate(energies_rayleigh)
    energies_photoelectric = np.concatenate(energies_photoelectric)

    # energy_histogram([np.array(electron_energy_deposition)])
    energy_histogram([np.array(electron_energy_deposition), np.array(electron_energy_from_compton), np.array(electron_energy_from_photoelectric)],
                     ['Total', 'From Compton', 'From photoelectric'])

    x_initial, y_initial, z_initial = np.array(x_initial_list), np.array(y_initial_list), np.array(z_initial_list)
    idx_initial = np.array(idx_initial_list)

    dims = [2, 24, 7, 3570]
    gantry_id_initial, rsector_id_initial, crystal_id_initial, layer_id_initial = np.unravel_index(idx_initial, dims)
    selection = (gantry_id_initial == 0) & (rsector_id_initial == 0) & (crystal_id_initial == 0)
    selection = np.ones(gantry_id_initial.shape, dtype=bool)

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    # ax.scatter(x_initial[selection], y_initial[selection], z_initial[selection])
    ax.scatter(x_initial, y_initial)
    # ax.set_aspect('equal')
    for angle in np.arange(0, 360, 15):
        add_boxes(ax, [6, 30], [0.6, 0], [16, 1], 429.799, angle)
        add_boxes(ax, [6, 30], [0.6, 0], [16, 1], 463.399, angle)

    for angle in (np.arange(0, 360, 36) + 18):
        add_boxes(ax, [6, 30], [0.6, 0], [16, 1], 190, angle)

    for ii in range(len(electron_path_x_compton)):
        ax.plot(electron_path_x_compton[ii], electron_path_y_compton[ii], color='tab:green')

    for ii in range(len(electron_path_x_photoelectric)):
        ax.plot(electron_path_x_photoelectric[ii], electron_path_y_photoelectric[ii], color='tab:red')

    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')
    # ax.set_xlim(-80, 80)
    # ax.set_ylim(400, 480)
    plt.show()

    electron_travel_distance = np.array(electron_travel_distance)

    distance_edges = np.geomspace(1e-4, 1e3, 100)
    distance_centers = (distance_edges[1:] + distance_edges[:-1]) / 2
    distance_width = distance_edges[1:] - distance_edges[:-1]

    h, _ = np.histogram(electron_travel_distance, bins=distance_edges)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.bar(distance_centers, h, width=distance_width)
    ax.set_xscale('log')
    ax.set_xlabel('Electron distance [mm]')
    ax.set_ylabel('Count')
    plt.show()

    # np.savez(sys.path[0] + '/data/metadata_secondaries.npz',
    #          step_count=photons_step_count,
    #          compton_count=compton_count,
    #          rayleigh_count=rayleigh_count,
    #          photoelectric_count=photoelectric_count,
    #          is_parent=is_parent,
    #          energies_initial_compton=energies_initial_compton,
    #          energies_deposited_compton=energies_deposited_compton,
    #          energies_rayleigh=energies_rayleigh,
    #          energies_photoelectric=energies_photoelectric)

    distance_to_closest_interaction = np.array(distance_to_closest_interaction)
    process_of_closest_interaction = np.array(process_of_closest_interaction)
    electron_parents = np.array(electron_parents)

    print(np.bincount(process_of_closest_interaction))
    print(np.sum(np.bincount(process_of_closest_interaction)))
    print(orphan_electron_counter)
    print(tertiary_electron_counter)
    print(tertiary_photon_counter)


    # distance_edges = np.linspace(0, 100, 101)
    distance_edges = np.geomspace(1e-4, 1e4, 101)
    distance_centers = (distance_edges[1:] + distance_edges[:-1]) / 2
    distance_width = distance_edges[1:] - distance_edges[:-1]

    h_0, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 0], bins=distance_edges)
    h_2, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 2], bins=distance_edges)
    h_6, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 6], bins=distance_edges)

    # h_1, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 1], bins=distance_edges)
    h_4, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 4], bins=distance_edges)
    # h_5, _ = np.histogram(distance_to_closest_interaction[process_of_closest_interaction == 5], bins=distance_edges)

    print(np.sum(h_2[distance_centers < 0.2]))
    print(np.sum(h_6[distance_centers < 0.2]))

    # plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots()
    # ax.bar(distance_centers, h_0, width=distance_width, alpha=0.75, label='Rayleigh')
    # ax.bar(distance_centers, h_2, width=distance_width, alpha=0.75, label='Compton')
    # ax.bar(distance_centers, h_6, width=distance_width, alpha=0.75, label='Photoelectric')
    # # ax.bar(distance_centers, h_1, width=distance_width, alpha=0.5, label='1')
    # # ax.bar(distance_centers, h_4, width=distance_width, alpha=0.5, label='4')
    # # ax.bar(distance_centers, h_5, width=distance_width, alpha=0.5, label='5')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel('Distance to closest interaction [mm]')
    # ax.set_ylabel('Count')
    # ax.legend()
    # plt.show()


    electrons_step_count = np.array(electrons_step_count)
    transportation_count = np.array(transportation_count)
    bremsstrahlung_count = np.array(bremsstrahlung_count)
    ionization_count = np.array(ionization_count)
    msc_count = np.array(msc_count)

    electron_parents = np.array(electron_parents)


    # print(np.sum(np.abs(photons_step_count - compton_count - rayleigh_count - photoelectric_count)))
    # print(np.sum(np.abs(electrons_step_count - transportation_count - bremsstrahlung_count - ionization_count - msc_count)))

    # print(np.sum(ionization_count))

    h = np.bincount(electrons_step_count)
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.bar(np.arange(h.size), h, width=1)
    ax.set_xlim(0, 200)
    # ax.set_yscale('log')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Count')
    plt.show()

    return 0


def add_boxes(ax, dimensions, spacings, n_rep, radius, angle, edge_color='tab:orange'):
    x_centers = (np.arange(n_rep[0]) - (n_rep[0] - 1) / 2) * (dimensions[0] + spacings[0]) - dimensions[0] / 2
    y_centers = - dimensions[1] / 2 + radius

    for ii in range(n_rep[0]):
        ax.add_patch(Rectangle(xy=(x_centers[ii], y_centers), width=dimensions[0], height=dimensions[1],
                               rotation_point=(0, 0), angle=angle, facecolor='none', edgecolor=edge_color))

    return 0


if __name__ == '__main__':
    main()
