"""
Read root files

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import isfile, basename
from os import sep
import numpy as np
from uproot import open
from scipy.sparse import csr_array
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


def main():
    """"""
    # Gate output
    root_file_name = '/home/martin/J-PET/Gate_mac_9.0/TB_J-PET_Brain/Output/2023/results_1MBq_10s_1.root'
    root_file = open(root_file_name)
    singles_tb = root_file['SinglesNoWLS']
    singles_b = root_file['B_Singles']
    # check_delta_t(singles_tb, ['rsectorID', 'crystalID', 'layerID'], [24, 7, 3520])
    # check_delta_t(singles_b, ['rsectorID', 'crystalID'], [10, 1760])
    print(singles_tb.num_entries + singles_b.num_entries)

    # myTreeMerger.cpp output
    # merged_root_file = open('/home/martin/J-PET/TB_J-PET_Brain/Sensitivity/MergedSingles0.root')
    # print(merged_root_file.keys())
    # merged_singles = merged_root_file['Singles']
    # print(merged_singles.num_entries)
    # print(merged_singles.keys())

    # # makeSinglesBrain.cpp output
    # brain_root_file = open('/home/martin/J-PET/TB_J-PET_Brain/Sensitivity/Singles0.root')
    # # print(brain_root_file.keys())
    # singles_brain = brain_root_file['Singles']
    # print(singles_brain.num_entries)

    """"""
    # Get the hits
    hits = root_file['Hits']

    # In the TB module, the layerID runs through (16 x 110) + 50 + (16 x 110) values, while the crystal ID does the
    # equivalent for the brain module, hence adjusting the indexing here
    # todo: Fix in Gate
    gantry_id_hits, rsector_id_hits, crystal_id_hits, layer_id_hits = hits_indexing(
        np.array(hits['gantryID']), np.array(hits['rsectorID']), np.array(hits['crystalID']), np.array(hits['layerID']))

    # De-cypher the linear layerID index
    layer_linear_to_subscript_indexing(layer_id_hits,
                                       gantry_id=gantry_id_hits, rsector_id=rsector_id_hits, crystal_id=crystal_id_hits,
                                       pos_x=np.array(hits['posX']), pos_y=np.array(hits['posY']), pos_z=np.array(hits['posZ']))
    sys.exit()

    # Change the positional index to a linear one
    dims = [2, 24, 7, 3570]
    print('Linearizing the positional index ...')
    pos_idx_hits = np.ravel_multi_index((gantry_id_hits, rsector_id_hits, crystal_id_hits, layer_id_hits), dims)

    # todo: Find a better way to read the number of events processed
    n_event_ids = np.round(root_file['total_nb_primaries'].all_members['fTsumwx']).astype(int)
    print(n_event_ids)

    # todo: Build a processID
    process_name = np.array(hits['processName'])
    process_id = process_statistics(process_name, show_histogram=False)
    sys.exit()

    (entry_indices_paths, pos_idx_paths, pos_x_paths, pos_y_paths, pos_z_paths, time_paths, e_dep_paths, process_id_paths,
     eid_paths, tid_paths, pid_paths, pdg_paths) = load_or_get_paths(hits, pos_idx_hits, process_id, root_file_name)

    # print(pdg_paths.size)
    # pdg_histogram(pdg_paths)

    # Verify
    # b = np.array([pos_idx_paths[ii].size for ii in range(pos_idx_paths.size)])
    # print(pos_idx_hits.size)
    # print(b)
    # print(np.sum(b))

    # Initial (and final) indices with shared eventID
    eid_idx = np.argwhere(np.diff(eid_paths) > 0)[:, 0]
    eid_idx = np.insert(eid_idx + 1, 0, 0)
    eid_idx = np.append(eid_idx, eid_paths.size)
    entry_indices = np.arange(eid_paths.size)

    secondary_count = 0
    secondaries = np.zeros(eid_paths.shape, dtype=bool)

    # Loop over events
    for ii in tqdm(range(eid_idx.size - 1)):
        entry_indices_temp = entry_indices[eid_idx[ii]:eid_idx[ii + 1]]
        track_id_temp = tid_paths[entry_indices_temp]
        pdg_temp = pdg_paths[entry_indices_temp]

        if np.max(track_id_temp) > 2:
            secondary_count += 1
            secondaries[entry_indices_temp] = True

        # Loop over tracks
        for jj in entry_indices_temp:
            pass

    print(secondary_count)
    print(secondary_count / (eid_idx.size - 1) * 100)

    secondaries = ~ secondaries.copy()

    # Save the events with secondaries in separate file
    np.savez(sys.path[0] + '/data/paths_1MBq_10s_1_primaries_only.npz',
             entry_indices_paths=entry_indices_paths[secondaries],
             pos_idx_paths=pos_idx_paths[secondaries],
             pos_x_paths=pos_x_paths[secondaries],
             pos_y_paths=pos_y_paths[secondaries],
             pos_z_paths=pos_z_paths[secondaries],
             time_paths=time_paths[secondaries],
             e_dep_paths=e_dep_paths[secondaries],
             process_id_paths=process_id_paths[secondaries],
             event_id_paths=eid_paths[secondaries],
             track_id_paths=tid_paths[secondaries],
             parent_track_id_paths=pid_paths[secondaries],
             pdg_encoding_paths=pdg_paths[secondaries])

    # print(np.bincount(np.array(a, dtype=int)))
    # print(np.bincount(np.concatenate(c)))



    sys.exit()

    # todo: Find the associated initial gammas
    # todo: Can an electron lead to another electron

    aa = np.ravel_multi_index((eid_paths, pid_paths), (n_event_ids, 172))
    bb = np.ravel_multi_index((event_id, track_id), (n_event_ids, 172))

    sorter = np.argsort(bb)
    gg = sorter[np.searchsorted(bb, aa, sorter=sorter, side='left')]
    ll = pdg_encoding[gg]

    # for ii in range(eid_paths.size):
    for ii in range(1000):
        print(np.sum(e_dep_paths[ii]))
        print(tid_paths[ii])
        selection = (event_id == eid_paths[ii]) & (track_id == pid_paths[ii])
        # selection = (event_id == eid_paths[ii]) & (track_id == tid_paths[ii])
        print(pdg_encoding[selection])
        print(np.sum(e_dep[selection]))
        print()







    # compare to brute force

    print(np.sum(ll == 11) / ll.size)
    print(np.sum(ll == 22) / ll.size)
    print(pdg_encoding[gg][:100])
    # sys.exit()
    #
    # print(aa)
    # print(bb)
    #
    # sys.exit()
    # print(np.min(pid_paths))
    # print(np.max(pid_paths))
    # sys.exit()

    print(eid_paths[:100])
    # print(pos_x_paths[0])
    print(np.sum(e_dep_paths[16]))
    # print(pdg_encoding[event_id == 7])
    # print(e_dep[event_id == 7])
    # print(process_name[event_id == 7])
    fig, ax = plt.subplots()
    ax.plot(e_dep_paths[16])
    plt.show()
    sys.exit()

    dp = np.sqrt(np.array([0 * (pos_x_paths[ii][-1] - pos_x_paths[ii][0]) ** 2 +
                           0 * (pos_y_paths[ii][-1] - pos_y_paths[ii][0]) ** 2 +
                           1 * (pos_z_paths[ii][-1] - pos_z_paths[ii][0]) ** 2 for ii in range(pos_z_paths.size)]))

    # bin_edges = np.linspace(0, 100, 100)
    bin_edges = np.geomspace(1e-4, 1e3, 100)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    h, _ = np.histogram(dp, bins=bin_edges)

    # h_normalized = h / np.trapz(h, x=bin_centers)
    h_normalized = h / np.sum(h)

    intersection_probability = np.minimum(bin_centers / 3, 1)
    # p = np.trapz(intersection_probability * h_normalized, x=bin_centers)
    p = np.sum(intersection_probability * h_normalized)

    print(p)

    fig, ax = plt.subplots()
    ax.bar(bin_centers, h_normalized, width=bin_width)
    ax_twin = ax.twinx()
    ax_twin.plot(bin_centers, intersection_probability, color='tab:orange')
    ax_twin.set_ylim(bottom=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta z$ [mm]')
    plt.show()

    print(np.min(dp))
    print(np.max(dp))
    print(np.mean(dp))
    print(np.median(dp))
    sys.exit()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for ii in range(10):
        ax.plot(pos_x_paths[ii], pos_y_paths[ii], pos_z_paths[ii])
    plt.show()

    sys.exit()

    # print(np.max(track_id[pdg_encoding == 22]))


    # b = 20823628 / 2056454
    event_id_hits = np.array(hits['eventID'])
    photon_id_hits = np.array(hits['photonID'])

    n_crystal_compton = np.array(hits['nCrystalCompton'])
    print(np.max(n_crystal_compton))
    print(np.max(photon_id_hits))
    print(n_crystal_compton)

    sys.exit()

    # Construct another index consisting of the eventID and the positional index
    sup_idx_hits = np.ravel_multi_index((event_id_hits, pos_idx_hits), (n_event_ids, np.prod(dims)))

    # Reduce the index, since most of the entries are empty
    sub_idx_hits_unique, sub_idx_hits_index, sup_idx_hits_reduced = np.unique(sup_idx_hits, return_index=True, return_inverse=True)

    # Check if photons from different IDs are merged
    h_ph = np.bincount(sup_idx_hits_reduced, weights=photon_id_hits) / np.bincount(sup_idx_hits_reduced)

    # If different photon IDs contribute, then h_ph is non-integer
    g_ph = h_ph - np.round(h_ph)

    # Determine, which indices of the reduced array have multiple photonIDs in the same volume
    sup_idx_hits_reduced_multiple = np.arange(h_ph.size)[g_ph != 0]

    # Go back to the sub-index
    sub_idx_hits_multiple = sub_idx_hits_unique[sup_idx_hits_reduced_multiple]

    # Extract the eventIDs with multiple photonIDs within a layer
    event_id_hits_multiple, _ = np.unravel_index(sub_idx_hits_multiple, (n_event_ids, np.prod(dims)))

    # Visualize
    # show_merged_hits_with_multiple_photon_ids(hits, event_id_hits_multiple)
    # sys.exit()

    # sup_idx_hits = np.ravel_multi_index((np.array(hits['photonID']) - 1, np.array(hits['eventID']), pos_idx_hits), (2, n_event_ids, np.prod(dims)))
    # _, sup_idx_hits = np.unique(sup_idx_hits, return_inverse=True)
    # h_ph = np.bincount(sup_idx_hits)
    # print(np.sum(h_ph > 1))

    # Identify the ones to be ditched by volume and eventID

    """"""
    merge_singles(singles_tb, singles_b, event_id_hits_multiple, sub_idx_hits_unique, photon_id_hits[sub_idx_hits_index], dims, n_event_ids)

    # print(np.unique(np.array(singles_tb['eventID'])).size)  # Gets that number (not that number)

    return 0


def check_delta_t(singles, keys, dims):
    # Read the IDs
    ids = []
    for ii in range(len(keys)):
        ids.append(np.array(singles[keys[ii]]))

    # Read the time
    time = np.array(singles['time'])

    # Change to linear index
    n_dims = np.prod(dims)
    lin_idx = np.ravel_multi_index(ids, dims)

    # Histogram
    h_idx = np.bincount(lin_idx, minlength=n_dims)

    # Layer indices with more than one interaction
    idx_g1 = np.arange(n_dims)[h_idx > 1]

    # Calculate the mutual time differences
    time_differences = []
    for ii in tqdm(range(idx_g1.size)):
        times = time[lin_idx == idx_g1[ii]][:, None]
        differences = times - times.T
        indices = np.tril_indices(times.size, k=-1)
        time_differences.append(differences[indices])
    time_differences_tot = np.concatenate(time_differences)

    # Show histogram
    bin_edges = np.geomspace(1e-10, 1e0, 100)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    h_dt, _ = np.histogram(time_differences_tot, bins=bin_edges)

    fig, ax = plt.subplots()
    ax.bar(bin_centers, h_dt, width=bin_width)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

    return 0


def hits_indexing(gantry_id, rsector_id, crystal_id, layer_id):
    print('Adjusting the positional indexing of the hits ...')
    # Indices of the total body module
    idx_tb = gantry_id == 0

    # Indices of the brain module
    idx_b = gantry_id == 1

    # Allocate
    gantry_id_return = gantry_id.copy()
    rsector_id_return = rsector_id.copy()
    crystal_id_return = np.zeros(crystal_id.shape, dtype=np.int32)
    layer_id_return = np.zeros(layer_id.shape, dtype=np.int32)

    # Leave the TB indices unchanged
    crystal_id_return[idx_tb] = crystal_id[idx_tb]
    layer_id_return[idx_tb] = layer_id[idx_tb]

    # Flip the brain indices
    crystal_id_return[idx_b] = layer_id[idx_b] + 1
    layer_id_return[idx_b] = crystal_id[idx_b]

    return gantry_id_return, rsector_id_return, crystal_id_return, layer_id_return


def layer_linear_to_subscript_indexing(layer_id, gantry_id=None, rsector_id=None, crystal_id=None, pos_x=None, pos_y=None, pos_z=None):
    # Scintillators
    # Y: 16
    # Z: 110

    # WLS
    # Z: 50

    y_rep = 16
    z_rep = 110

    layer_1 = layer_id < (y_rep * z_rep)
    layer_2 = (layer_id >= (y_rep * z_rep)) & (layer_id < (2 * y_rep * z_rep))
    layer_3 = layer_id >= (2 * y_rep * z_rep)

    x_idx_1, y_idx_1, z_idx_1 = np.unravel_index(layer_id[layer_1], (1, 16, 110), order='F')
    x_idx_2, y_idx_2, z_idx_2 = np.unravel_index(layer_id[layer_2] - (y_rep * z_rep), (1, 16, 110), order='F')
    x_idx_3, y_idx_3, z_idx_3 = np.unravel_index(layer_id[layer_3] - (2 * y_rep * z_rep), (1, 1, 50), order='F')

    x_idx = np.zeros(layer_id.shape, dtype=layer_id.dtype)
    x_idx[layer_1] = x_idx_1
    x_idx[layer_2] = x_idx_2 + 1
    x_idx[layer_3] = x_idx_3 + 2

    y_idx = np.zeros(layer_id.shape, dtype=layer_id.dtype)
    y_idx[layer_1] = y_idx_1
    y_idx[layer_2] = y_idx_2
    y_idx[layer_3] = y_idx_3

    z_idx = np.zeros(layer_id.shape, dtype=layer_id.dtype)
    z_idx[layer_1] = z_idx_1
    z_idx[layer_2] = z_idx_2
    z_idx[layer_3] = z_idx_3

    if (gantry_id is not None) and (rsector_id is not None) and (crystal_id is not None) and (pos_x is not None) and (pos_y is not None) and (pos_z is not None):
        # Select a single module to plot
        #          0 ... 1            0 ... 23            0 ... 6
        sub_set = (gantry_id == 0) & (rsector_id == 0) & (crystal_id == 0)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(pos_x[sub_set], pos_y[sub_set], pos_z[sub_set], c=layer_id[sub_set])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(pos_x[sub_set], pos_y[sub_set], pos_z[sub_set], c=x_idx[sub_set])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        plt.show()

    return x_idx, y_idx, z_idx


def process_statistics(process_name, show_histogram=False):
    # Get the histogram
    processes, process_counts = np.unique(process_name, return_counts=True)

    # Cast to processID
    process_id = np.zeros(process_name.shape, dtype=int)
    for ii in range(processes.size):
        process_id[process_name == processes[ii]] = ii
        print('%i | %s' % (ii, processes[ii]))

    # # Check consistency
    # print(process_counts)
    # print(np.bincount(process_id))

    if show_histogram:
        plt.rcParams.update({'font.size': 16})
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax0.bar(np.arange(processes.size), process_counts)
        ax0.set_xticks(np.arange(processes.size))
        ax0.set_xticklabels(processes, rotation=45, ha='right', rotation_mode='anchor')
        ax1.bar(np.arange(processes.size), process_counts)
        ax1.set_xticks(np.arange(processes.size))
        ax1.set_xticklabels(processes, rotation=45, ha='right', rotation_mode='anchor')
        ax1.set_yscale('log')
        fig.tight_layout()
        plt.show()

    return process_id


def merge_singles(singles_tb, singles_b, event_id_multiple, sup_idx_map, photon_id_map, dims, n_event_ids):
    # Read all TB singles entries
    (run_id_tb, event_id_tb, source_id_tb, source_pos_x_tb, source_pos_y_tb, source_pos_z_tb,
     time_tb, energy_tb, global_pos_x_tb, global_pos_y_tb, global_pos_z_tb,
     gantry_id_tb, rsector_id_tb, module_id_tb, submodule_id_tb, crystal_id_tb, layer_id_tb,
     compton_phantom_tb, compton_crystal_tb, rayleigh_phantom_tb, rayleigh_crystal_tb,
     axial_pos_tb, rotation_angle_tb, compt_vol_name_tb, rayleigh_vol_name_tb) = read_singles_entries(singles_tb)

    # Read all brain singles entries
    (run_id_b, event_id_b, source_id_b, source_pos_x_b, source_pos_y_b, source_pos_z_b,
     time_b, energy_b, global_pos_x_b, global_pos_y_b, global_pos_z_b,
     gantry_id_b, rsector_id_b, module_id_b, submodule_id_b, crystal_id_b, layer_id_b,
     compton_phantom_b, compton_crystal_b, rayleigh_phantom_b, rayleigh_crystal_b,
     axial_pos_b, rotation_angle_b, compt_vol_name_b, rayleigh_vol_name_b) = read_singles_entries(singles_b)

    # Define the data type
    dtype_singles = np.dtype([('runID', np.int32),              # 1 x 4 bytes
                              ('eventID', np.int32),            # 1 x 4 bytes
                              ('photonID', np.int32),           # 1 x 4 bytes (added)
                              ('sourceID', np.int32),           # 1 x 4 bytes
                              ('sourcePos', np.float32, (3,)),  # 3 x 4 bytes
                              ('time', np.float64),             # 1 x 8 bytes
                              ('energy', np.float32),           # 1 x 4 bytes
                              ('globalPos', np.float32, (3,)),  # 3 x 4 bytes
                              ('gantryID', np.int32),           # 1 x 4 bytes
                              ('rsectorID', np.int32),          # 1 x 4 bytes
                              ('moduleID', np.int32),           # 1 x 4 bytes
                              ('submoduleID', np.int32),        # 1 x 4 bytes
                              ('crystalID', np.int32),          # 1 x 4 bytes
                              ('layerID', np.int32),            # 1 x 4 bytes
                              ('comptonPhantom', np.int32),     # 1 x 4 bytes
                              ('comptonCrystal', np.int32),     # 1 x 4 bytes
                              ('RayleighPhantom', np.int32),    # 1 x 4 bytes
                              ('RayleighCrystal', np.int32),    # 1 x 4 bytes
                              ('axialPos', np.float32),         # 1 x 4 bytes
                              ('rotationAngle', np.float32),    # 1 x 4 bytes
                              ('comptVolName', object),         # 1 x 8 bytes
                              ('RayleighVolName', object),      # 1 x 8 bytes
                              ])
    #                                                           -------------
    #                                                           116 bytes per entry
    # print(dtype_data.itemsize)

    # Flip the crystalID and layerID for the brain module
    temp = crystal_id_b.copy()
    crystal_id_b = layer_id_b.copy() + 1
    layer_id_b = temp.copy()

    # Allocate the structured array
    singles_array = np.empty(singles_tb.num_entries + singles_b.num_entries, dtype=dtype_singles)

    # Write into it
    singles_array['runID'] = np.hstack((run_id_tb, run_id_b))
    singles_array['eventID'] = np.hstack((event_id_tb, event_id_b))
    singles_array['sourceID'] = np.hstack((source_id_tb, source_id_b))
    singles_array['sourcePos'] = np.vstack((np.hstack((source_pos_x_tb, source_pos_x_b)),
                                            np.hstack((source_pos_y_tb, source_pos_y_b)),
                                            np.hstack((source_pos_z_tb, source_pos_z_b)))).T
    singles_array['time'] = np.hstack((time_tb, time_b))
    singles_array['energy'] = np.hstack((energy_tb, energy_b))
    singles_array['globalPos'] = np.vstack((np.hstack((global_pos_x_tb, global_pos_x_b)),
                                            np.hstack((global_pos_y_tb, global_pos_y_b)),
                                            np.hstack((global_pos_z_tb, global_pos_z_b)))).T
    singles_array['gantryID'] = np.hstack((gantry_id_tb, gantry_id_b))
    singles_array['rsectorID'] = np.hstack((rsector_id_tb, rsector_id_b))
    singles_array['moduleID'] = np.hstack((module_id_tb, module_id_b))
    singles_array['submoduleID'] = np.hstack((submodule_id_tb, submodule_id_b))
    singles_array['crystalID'] = np.hstack((crystal_id_tb, crystal_id_b))
    singles_array['layerID'] = np.hstack((layer_id_tb, layer_id_b))
    singles_array['comptonPhantom'] = np.hstack((compton_phantom_tb, compton_phantom_b))
    singles_array['comptonCrystal'] = np.hstack((compton_crystal_tb, compton_crystal_b))
    singles_array['RayleighPhantom'] = np.hstack((rayleigh_phantom_tb, rayleigh_phantom_b))
    singles_array['RayleighCrystal'] = np.hstack((rayleigh_crystal_tb, rayleigh_crystal_b))
    singles_array['axialPos'] = np.hstack((axial_pos_tb, axial_pos_b))
    singles_array['rotationAngle'] = np.hstack((rotation_angle_tb, rotation_angle_b))
    singles_array['comptVolName'] = np.hstack((compt_vol_name_tb, compt_vol_name_b))
    singles_array['RayleighVolName'] = np.hstack((rayleigh_vol_name_tb, rayleigh_vol_name_b))

    # Remove the entries with eventIDs with multiple photonIDs
    event_id = singles_array['eventID']
    event_id_bool = np.zeros(event_id.shape, dtype=bool)
    for ii in event_id_multiple:
        event_id_bool = event_id_bool | (event_id == ii)

    singles_array = singles_array[~event_id_bool]

    # Calculate the sup index
    pos_idx_singles = np.ravel_multi_index((singles_array['gantryID'], singles_array['rsectorID'], singles_array['crystalID'], singles_array['layerID']), dims)
    sup_idx_singles = np.ravel_multi_index((singles_array['eventID'], pos_idx_singles), (n_event_ids, np.prod(dims)))

    # Determine, where in the map are the sup_idx_singles
    insertion_indices = np.searchsorted(sup_idx_map, sup_idx_singles)

    # Assign a photon ID to each single
    singles_array['photonID'] = photon_id_map[insertion_indices]

    # np.save(sys.path[0] + '/data/singles_1.npy', singles_array)

    return 0


def read_singles_entries(singles):
    run_id = np.array(singles['runID'])
    event_id = np.array(singles['eventID'])
    source_id = np.array(singles['sourceID'])

    source_pos_x = np.array(singles['sourcePosX'])
    source_pos_y = np.array(singles['sourcePosY'])
    source_pos_z = np.array(singles['sourcePosZ'])

    time = np.array(singles['time'])
    energy = np.array(singles['energy'])

    global_pos_x = np.array(singles['globalPosX'])
    global_pos_y = np.array(singles['globalPosY'])
    global_pos_z = np.array(singles['globalPosZ'])

    gantry_id = np.array(singles['gantryID'])
    rsector_id = np.array(singles['rsectorID'])
    module_id = np.array(singles['moduleID'])
    submodule_id = np.array(singles['submoduleID'])
    crystal_id = np.array(singles['crystalID'])
    layer_id = np.array(singles['layerID'])

    compton_phantom = np.array(singles['comptonPhantom'])
    compton_crystal = np.array(singles['comptonCrystal'])
    rayleigh_phantom = np.array(singles['RayleighPhantom'])
    rayleigh_crystal = np.array(singles['RayleighCrystal'])

    axial_pos = np.array(singles['axialPos'])
    rotation_angle = np.array(singles['rotationAngle'])

    compt_vol_name = np.array(singles['comptVolName'])
    rayleigh_vol_name = np.array(singles['RayleighVolName'])

    return (run_id, event_id, source_id, source_pos_x, source_pos_y, source_pos_z,
            time, energy, global_pos_x, global_pos_y, global_pos_z,
            gantry_id, rsector_id, module_id, submodule_id, crystal_id, layer_id,
            compton_phantom, compton_crystal, rayleigh_phantom, rayleigh_crystal,
            axial_pos, rotation_angle, compt_vol_name, rayleigh_vol_name)


def show_merged_hits_with_multiple_photon_ids(hits, event_id_hits_multiple):
    # Extract
    event_id_hits = np.array(hits['eventID'])
    photon_id_hits = np.array(hits['photonID'])

    pos_x_hits = np.array(hits['posX'])
    pos_y_hits = np.array(hits['posY'])
    pos_z_hits = np.array(hits['posZ'])

    source_x_hits = np.array(hits['sourcePosX'])
    source_y_hits = np.array(hits['sourcePosY'])
    source_z_hits = np.array(hits['sourcePosZ'])

    time_hits = np.array(hits['time'])

    # Plot
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # for ii in range(event_id_hits_multiple.size):
    for ii in [1]:
        #
        selector_1 = (event_id_hits == event_id_hits_multiple[ii]) & (photon_id_hits == 1)
        selector_2 = (event_id_hits == event_id_hits_multiple[ii]) & (photon_id_hits == 2)

        pos_x_1, pos_x_2 = pos_x_hits[selector_1], pos_x_hits[selector_2]
        pos_y_1, pos_y_2 = pos_y_hits[selector_1], pos_y_hits[selector_2]
        pos_z_1, pos_z_2 = pos_z_hits[selector_1], pos_z_hits[selector_2]

        source_pos_x_1, source_pos_x_2 = source_x_hits[selector_1], source_x_hits[selector_2]
        source_pos_y_1, source_pos_y_2 = source_y_hits[selector_1], source_y_hits[selector_2]
        source_pos_z_1, source_pos_z_2 = source_z_hits[selector_1], source_z_hits[selector_2]

        # time_1, time_2 = time_hits[selector_1], time_hits[selector_2]

        x_1 = np.insert(pos_x_1, 0, source_pos_x_1[0])
        y_1 = np.insert(pos_y_1, 0, source_pos_y_1[0])
        z_1 = np.insert(pos_z_1, 0, source_pos_z_1[0])

        x_2 = np.insert(pos_x_2, 0, source_pos_x_2[0])
        y_2 = np.insert(pos_y_2, 0, source_pos_y_2[0])
        z_2 = np.insert(pos_z_2, 0, source_pos_z_2[0])

        ax.plot(x_1, y_1, z_1, color='tab:blue', label=r'$\gamma_1$')
        ax.plot(x_2, y_2, z_2, color='tab:orange', label=r'$\gamma_2$')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        ax.legend()

    plt.show()

    return 0


def load_or_get_paths(hits, pos_idx, process_id, root_file_name):
    # Output file
    # paths_file = sys.path[0] + '/data/paths' + basename(root_file_name)[7:-5] + '.npz'
    paths_file = sys.path[0] + '/data/paths_' + root_file_name.split(sep)[-2] + '.npz'

    # If it has already been created, load
    if isfile(paths_file):
        print('Loading the paths ...')
        npz_file = np.load(paths_file, allow_pickle=True)
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

    # Otherwise, create and save
    else:
        # Load the data from the hits
        pos_x = np.array(hits['posX'])
        pos_y = np.array(hits['posY'])
        pos_z = np.array(hits['posZ'])
        time = np.array(hits['time'])
        e_dep = np.array(hits['edep'])
        event_id = np.array(hits['eventID'])
        track_id = np.array(hits['trackID'])
        parent_track_id = np.array(hits['parentID'])
        # photon_id = np.array(hits['photonID'])
        pdg_encoding = np.array(hits['PDGEncoding'])
        # pdg_histogram(pdg_encoding)

        print('Getting the paths and saving to\n"%s"' % paths_file)
        # Get the paths
        entry_indices_paths, pos_idx_paths, pos_x_paths, pos_y_paths, pos_z_paths, time_paths, e_dep_paths, process_id_paths, \
            eid_paths, tid_paths, pid_paths, pdg_paths = get_paths(pos_idx, pos_x, pos_y, pos_z, time, e_dep, event_id,
                                                                   track_id, parent_track_id, pdg_encoding, process_id)

        # Save the result
        np.savez(paths_file,
                 entry_indices_paths=entry_indices_paths,
                 pos_idx_paths=pos_idx_paths,
                 pos_x_paths=pos_x_paths,
                 pos_y_paths=pos_y_paths,
                 pos_z_paths=pos_z_paths,
                 time_paths=time_paths,
                 e_dep_paths=e_dep_paths,
                 process_id_paths=process_id_paths,
                 event_id_paths=eid_paths,
                 track_id_paths=tid_paths,
                 parent_track_id_paths=pid_paths,
                 pdg_encoding_paths=pdg_paths)

    return entry_indices_paths, pos_idx_paths, pos_x_paths, pos_y_paths, pos_z_paths, time_paths, e_dep_paths, process_id_paths, \
        eid_paths, tid_paths, pid_paths, pdg_paths


def get_paths(pos_idx, pos_x, pos_y, pos_z, time, e_dep, event_id, track_id, parent_id, pdg_encoding, process_id, run_checks=True):
    # Allocate
    entry_indices_list = []
    pos_idx_paths_list = []
    pos_x_paths_list = []
    pos_y_paths_list = []
    pos_z_paths_list = []
    time_paths_list = []
    e_dep_paths_list = []
    process_id_paths_list = []
    event_id_list = []
    track_id_list = []
    parent_id_list = []
    pdg_encoding_list = []

    #
    event_id = fix_event_id(event_id)

    # Initial (and final) indices with shared eventID
    eid_idx = np.argwhere(np.diff(event_id) > 0)[:, 0]
    eid_idx = np.insert(eid_idx + 1, 0, 0)
    eid_idx = np.append(eid_idx, event_id.size)

    #
    entry_indices = np.arange(event_id.size)

    print('Collecting from %i events.' % (eid_idx.size - 1))
    for ii in tqdm(range(eid_idx.size - 1)):
        # Get the indices corresponding to the current eventID
        entry_indices_temp = entry_indices[eid_idx[ii]:eid_idx[ii + 1]]

        # Get the current eventID
        event_ids_temp = event_id[entry_indices_temp]

        if run_checks:
            # Check if only one eventID is processed
            if np.unique(event_ids_temp).size > 1:
                print(event_ids_temp)
                sys.exit('Error: not properly subdivided!')

        # Get the corresponding trackIDs
        track_ids_temp = track_id[entry_indices_temp]
        available_track_ids = np.unique(track_ids_temp)

        # print('Event %i: Getting %i tracks.' % (event_id_temp, available_track_ids.size))
        for track_id_temp in available_track_ids:
            selection = entry_indices_temp[track_ids_temp == track_id_temp]

            if run_checks:
                # Check if the trackID entries are sorted according to the time
                time_temp = time[selection]
                if not np.all(time_temp[:-1] <= time_temp[1:]):
                    print(time_temp)
                    sys.exit('Error: Not sorted!')

            entry_indices_list.append(entry_indices[selection])
            pos_idx_paths_list.append(pos_idx[selection])
            pos_x_paths_list.append(pos_x[selection])
            pos_y_paths_list.append(pos_y[selection])
            pos_z_paths_list.append(pos_z[selection])
            time_paths_list.append(time_temp)
            e_dep_paths_list.append(e_dep[selection])
            process_id_paths_list.append(process_id[selection])
            # print(e_dep[selection].size)

            parent_id_temp = parent_id[selection]
            pdg_encoding_temp = pdg_encoding[selection]

            if run_checks:
                if np.unique(parent_id_temp).size > 1:
                    print(parent_id_temp)
                    sys.exit('Error: not properly subdivided!')

                if np.unique(pdg_encoding_temp).size > 1:
                    print(pdg_encoding_temp)
                    sys.exit('Error: not properly subdivided!')

            event_id_list.append(event_ids_temp[0])
            track_id_list.append(track_id_temp)
            parent_id_list.append(parent_id_temp[0])
            pdg_encoding_list.append(pdg_encoding_temp[0])

    entry_indices_array = np.array(entry_indices_list, dtype=object)
    pos_idx_paths_array = np.array(pos_idx_paths_list, dtype=object)
    pos_x_paths_array = np.array(pos_x_paths_list, dtype=object)
    pos_y_paths_array = np.array(pos_y_paths_list, dtype=object)
    pos_z_paths_array = np.array(pos_z_paths_list, dtype=object)
    time_paths_array = np.array(time_paths_list, dtype=object)
    e_dep_paths_array = np.array(e_dep_paths_list, dtype=object)
    process_id_paths_array = np.array(process_id_paths_list, dtype=object)

    event_id_array = np.array(event_id_list)
    track_id_array = np.array(track_id_list)
    parent_id_array = np.array(parent_id_list)
    pdg_encoding_array = np.array(pdg_encoding_list)

    return (entry_indices_array, pos_idx_paths_array, pos_x_paths_array, pos_y_paths_array, pos_z_paths_array,
            time_paths_array, e_dep_paths_array, process_id_paths_array,
            event_id_array, track_id_array, parent_id_array, pdg_encoding_array)


def pdg_histogram(pdg_encoding):
    pdg_hist = np.bincount(pdg_encoding)
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(2.4, 4.8))
    ax.bar([11, 22], pdg_hist[pdg_hist > 0], width=10)
    ax.set_xticks([11, 22])
    ax.set_xlabel('PDG encoding')
    ax.set_ylabel('Count')
    plt.show()
    return 0


def fix_event_id(event_id):
    eid_idx = np.argwhere(np.diff(event_id) < 0)[:, 0]
    eid_idx = np.insert(eid_idx + 1, 0, 0)
    eid_idx = np.append(eid_idx, event_id.size)

    event_id_new = np.ones(event_id.shape, dtype=int)
    offset = 0

    for ii in range(eid_idx.size - 1):
        event_id_new[eid_idx[ii]:eid_idx[ii + 1]] = event_id[eid_idx[ii]:eid_idx[ii + 1]] + offset
        offset = event_id_new[eid_idx[ii + 1] - 1] + 1

    # fig, ax = plt.subplots()
    # ax.plot(event_id)
    # ax.plot(event_id_new, '--')
    # plt.show()

    return event_id_new


if __name__ == '__main__':
    main()
