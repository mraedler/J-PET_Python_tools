"""
Functions commonly used

@author: Martin Rädler
"""
# Python libraries
import sys
from glob import glob
from natsort import natsorted
from uproot import open as open_root
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array


def load_gate_data(time_resolution, phantom=False):
    # Time resolution set to 0, 200/sqrt(2), 400/sqrt(2), 600/sqrt(2) picoseconds
    if time_resolution == 0:
        idx = 0
    elif time_resolution == 200:
        idx = 1
    elif time_resolution == 400:
        idx = 2
    elif time_resolution == 600:
        idx = 3
    else:
        sys.exit('Error: time resolution unavailable.')

    new_tb_jpet_path = '/home/martin/J-PET/Gate_Output/New_TB-J-PET/'

    if phantom:
        simulation = ['2025-07-05_17-15-26', '2025-07-05_17-32-32', '2025-07-05_17-50-24', '2025-07-05_18-08-15']
    else:
        simulation = ['2025-07-05_15-53-07', '2025-07-05_16-16-25', '2025-07-05_16-39-04', '2025-07-05_16-58-44']

    root_files = natsorted(glob(new_tb_jpet_path + simulation[idx] + '/results_*.root'))

    necessary_keys = ['eventID1', 'sourceID1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'time1', 'energy1', 'globalPosX1', 'globalPosY1', 'globalPosZ1', 'gantryID1', 'rsectorID1', 'moduleID1', 'submoduleID1', 'crystalID1', 'layerID1', 'comptonCrystal1', 'RayleighCrystal1', 'comptonPhantom1', 'RayleighPhantom1',
                      'eventID2', 'sourceID2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'time2', 'energy2', 'globalPosX2', 'globalPosY2', 'globalPosZ2', 'gantryID2', 'rsectorID2', 'moduleID2', 'submoduleID2', 'crystalID2', 'layerID2', 'comptonCrystal2', 'RayleighCrystal2', 'comptonPhantom2', 'RayleighPhantom2']

    for ii in range(len(root_files)):
        # Load the simulation data
        root_file = open_root(root_files[ii])
        # for key, data_type in root_file['MergedCoincidences'].itertypenames():
        #     print(key)
        coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'], keys=necessary_keys, overwrite=False)

    return coincidences_struct


def filter_true(coincidences_struct, verbose=False):
    event_id_1, event_id_2 = coincidences_struct['eventID1'], coincidences_struct['eventID2']
    compton_crystal_1, compton_crystal_2 = coincidences_struct['comptonCrystal1'], coincidences_struct['comptonCrystal2']
    rayleigh_crystal_1, rayleigh_crystal_2 = coincidences_struct['RayleighCrystal1'], coincidences_struct['RayleighCrystal2']

    event_id_check = event_id_1 == event_id_2
    compton_scatter_check = (compton_crystal_1 == 1) & (compton_crystal_2 == 1)
    rayleigh_scatter_check = (rayleigh_crystal_1 == 0) & (rayleigh_crystal_2 == 0)

    true = event_id_check & compton_scatter_check & rayleigh_scatter_check

    if verbose:
        print('Passing eventID check: %1.2f %%.' % (np.sum(event_id_check) / event_id_check.size * 100))
        print('Passing Compton scatter check: %1.2f %%.' % (np.sum(compton_scatter_check) / compton_scatter_check.size * 100))
        print('Passing Rayleigh scatter check: %1.2f %%.' % (np.sum(rayleigh_scatter_check) / rayleigh_scatter_check.size * 100))
        print('Overall: %1.2f %%' % (np.sum(true) / true.size * 100))
    return true


def filter_phantom_scattered(coincidences_struct, verbose=False, vis=False):

    compton_phantom1, rayleigh_phantom1 = coincidences_struct['comptonPhantom1'], coincidences_struct['RayleighPhantom1']
    compton_phantom2, rayleigh_phantom2 = coincidences_struct['comptonPhantom2'], coincidences_struct['RayleighPhantom2']
    not_phantom_scattered = (compton_phantom1 == 0) & (rayleigh_phantom1 == 0) & (compton_phantom2 == 0) & (rayleigh_phantom2 == 0)

    if verbose:
        print('%1.2f %%' % (np.sum(not_phantom_scattered) / not_phantom_scattered.size * 100))

    if vis:
        compton_phantom1_stats, rayleigh_phantom1_stats = np.bincount(compton_phantom1), np.bincount(rayleigh_phantom1)
        compton_phantom2_stats, rayleigh_phantom2_stats = np.bincount(compton_phantom2), np.bincount(rayleigh_phantom2)

        fig, ax = plt.subplots()
        ax.bar(np.arange(compton_phantom1_stats.size), compton_phantom1_stats, alpha=0.75)
        ax.bar(np.arange(compton_phantom2_stats.size), compton_phantom2_stats, alpha=0.75)
        # ax.bar(np.arange(rayleigh_phantom1_stats.size), rayleigh_phantom1_stats, alpha=0.75)
        # ax.bar(np.arange(rayleigh_phantom2_stats.size), rayleigh_phantom2_stats, alpha=0.75)
        ax.set_yscale('log')
        plt.show()
        sys.exit()

    return not_phantom_scattered


def separate_into_detector_categories(gantry_id1, gantry_id2, verbose=False):
    tbtb = (gantry_id1 < 2) & (gantry_id2 < 2)
    tbbi = ((gantry_id1 < 2) & (gantry_id2 == 2)) | (gantry_id1 == 2) & (gantry_id2 < 2)
    bibi = (gantry_id1 == 2) & (gantry_id2 == 2)

    if verbose:
        n = gantry_id1.size
        print('TB-TB: %1.2f %%' % (np.sum(tbtb) / n * 100))
        print('TB-BI: %1.2f %%' % (np.sum(tbbi) / n * 100))
        print('BI-BI: %1.2f %%' % (np.sum(bibi) / n * 100))

    return tbtb, tbbi, bibi
