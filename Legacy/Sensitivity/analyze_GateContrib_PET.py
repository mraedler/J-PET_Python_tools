"""
Sensitivity analysis based on Singles

@author: Martin Rädler
"""
import sys

# Python libraries
import numpy as np
from uproot import open

# Auxiliary functions
from Gate_9_0.sensitivity_analysis import get_coincidence_indices, get_sensitivity_profile, plot_sensitivity_profile, plot_distance_distribution
from sensitivity_singles import interaction_statistics, energy_spectrum


def main():
    """
    Main function
    :return: 0
    """
    """Gate 9.0"""
    # Get the singles and construct a module index
    # time_0, energy_0, source_pos_0, global_pos_0, event_id_0, photon_id_0, gantry_id_0, rsector_id_0, compton_crystal_0, rayleigh_crystal_0 = (
    #     read_singles('/home/martin/J-PET/Gate_mac_9.0/GateContrib_PET/YourFile.root', 'Singles'))

    # e_dep_a, n_crystal_compton_a = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.0/505ff3c5/YourFile.root', ['Hits'])
    e_dep_a, n_crystal_compton_a = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.0/a27ef241/YourFile.root', ['Hits'])

    # e_dep_b, n_crystal_compton_b = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.3/6dc388a4/YourFile.root', ['Hits_LSO', 'Hits_BGO'])
    # e_dep_b, n_crystal_compton_b = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.3/29e2b2ae/YourFile.root', ['Hits_LSO', 'Hits_BGO'])
    # e_dep_b, n_crystal_compton_b = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.3/045ed344/YourFile.root', ['Hits_LSO', 'Hits_BGO'])
    # e_dep_b, n_crystal_compton_b = read_hits('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.3/6db928fd/YourFile.root', ['Hits_LSO', 'Hits_BGO'])
    e_dep_b, n_crystal_compton_b = read_hits('/home/martin/J-PET/Gate_mac_9.3/GateContrib_PET/YourFile.root', ['Hits_LSO', 'Hits_BGO'])
    # print(n_crystal_compton_b[n_crystal_compton_b < 0].size)
    # sys.exit()
    print(e_dep_b)
    # sys.exit()
    energy_spectrum([e_dep_a, e_dep_b], ['Gate_9.0', 'Gate_9.3'], 'Hits')
    interaction_statistics(n_crystal_compton_a, n_crystal_compton_b, 'comptonCrystal')
    sys.exit()

    time_0, energy_0, source_pos_0, global_pos_0, event_id_0, photon_id_0, gantry_id_0, rsector_id_0, compton_crystal_0, rayleigh_crystal_0 = (
        read_singles('/home/martin/J-PET/Gate_mac_bisect/GateContrib_PET_9.0/YourFile.root', 'Singles'))

    time_3, energy_3, source_pos_3, global_pos_3, event_id_3, photon_id_3, gantry_id_3, rsector_id_3, compton_crystal_3, rayleigh_crystal_3 = (
        read_singles('/home/martin/J-PET/Gate_mac_9.3/GateContrib_PET/YourFile.root', 'Singles_BGO'))

    energy_spectrum([energy_0, energy_3], ['Gate_9.0', 'Gate_9.3'], 'Singles')
    interaction_statistics(compton_crystal_0, compton_crystal_3, 'comptonCrystal')
    interaction_statistics(rayleigh_crystal_0, rayleigh_crystal_3, 'RayleighCrystal')
    sys.exit()
    # Get the coincidence indices
    idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton = (
        get_coincidence_indices(time_0, event_id_0, photon_id_0, rsector_id_0, compton_crystal_0, rayleigh_crystal_0, time_window=3e-9))

    # Get the coincidence indices
    idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton = (
        get_coincidence_indices(time_3, event_id_3, photon_id_3, rsector_id_3, compton_crystal_3, rayleigh_crystal_3, time_window=3e-9))


    return 0


def read_hits(root_file_name, hits_tags):
    root_file = open(root_file_name)
    print(root_file.keys())
    # sys.exit()

    e_dep = np.empty((0,), dtype=float)
    n_crystal_compton = np.empty((0,), dtype=int)

    for ii in range(len(hits_tags)):
        hits = root_file[hits_tags[ii]]
        e_dep = np.append(e_dep, np.array(hits['edep']))
        n_crystal_compton = np.append(n_crystal_compton, np.array(hits['nCrystalCompton']))

    return e_dep, n_crystal_compton


def read_singles(root_file_name, singles_tag):
    root_file = open(root_file_name)

    singles = root_file[singles_tag]
    time = np.array(singles['time'])
    energy = np.array(singles['energy'])

    source_x = np.array(singles['sourcePosX'])
    source_y = np.array(singles['sourcePosY'])
    source_z = np.array(singles['sourcePosZ'])
    source_pos = np.hstack((source_x[:, np.newaxis], source_y[:, np.newaxis], source_z[:, np.newaxis]))

    global_x = np.array(singles['globalPosX'])
    global_y = np.array(singles['globalPosY'])
    global_z = np.array(singles['globalPosZ'])
    global_pos = np.hstack((global_x[:, np.newaxis], global_y[:, np.newaxis], global_z[:, np.newaxis]))

    event_id = np.array(singles['eventID'])
    photon_id = np.arange(event_id.size)  # dummy photon ID
    gantry_id = np.array(singles['gantryID'])
    rsector_id = np.array(singles['rsectorID'])
    compton_crystal = np.array(singles['comptonCrystal'])
    rayleigh_crystal = np.array(singles['RayleighCrystal'])

    return time, energy, source_pos, global_pos, event_id, photon_id, gantry_id, rsector_id, compton_crystal, rayleigh_crystal


if __name__ == "__main__":
    main()
