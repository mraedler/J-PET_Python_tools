"""
Variable energy cut

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from uproot import open
from tqdm import tqdm
import matplotlib.pyplot as plt

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array
# from sensitivity_singles import energy_spectrum
from Legacy.Sensitivity.sensitivity_singles import energy_spectrum


def main():
    # overwrite = True
    overwrite = False

    # Load the coincidences including the 200 keV cut
    root_file = open('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_11-49-02/results.root')
    coincidences_cut = load_or_convert_to_structured_array(root_file['MergedCoincidences'], overwrite=overwrite,
                                                           keys=['energy1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonCrystal1',
                                                                 'energy2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'comptonCrystal2'])

    # Load the coincidences without the 200 keV cut
    root_file = open('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-03-06_19-05-29/results.root')
    coincidences_all = load_or_convert_to_structured_array(root_file['MergedCoincidences'], overwrite=overwrite,
                                                           keys=['energy1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonCrystal1',
                                                                 'energy2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'comptonCrystal2'])

    #
    energy_1 = coincidences_all['energy1']
    x_source_1 = coincidences_all['sourcePosX1']
    y_source_1 = coincidences_all['sourcePosY1']
    z_source_1 = coincidences_all['sourcePosZ1']
    compton_crystal_1 = coincidences_all['comptonCrystal1']

    energy_2 = coincidences_all['energy2']
    x_source_2 = coincidences_all['sourcePosX2']
    y_source_2 = coincidences_all['sourcePosY2']
    z_source_2 = coincidences_all['sourcePosZ2']
    compton_crystal_2 = coincidences_all['comptonCrystal2']

    # e_cut = np.linspace(0, 0.5, 51)
    # e_cut = np.linspace(0, 0.5, 501)
    e_cut = np.linspace(0, 0.5, 51)
    a = np.zeros(e_cut.shape)
    b = np.zeros(e_cut.shape)
    c = np.zeros(e_cut.shape)

    for ii in tqdm(range(e_cut.size)):
        energy_cut = (energy_1 >= e_cut[ii]) & (energy_2 >= e_cut[ii])
        x_source_1_temp, y_source_1_temp, z_source_1_temp, compton_crystal_1_temp = x_source_1[energy_cut], y_source_1[energy_cut], z_source_1[energy_cut], compton_crystal_1[energy_cut]
        x_source_2_temp, y_source_2_temp, z_source_2_temp, compton_crystal_2_temp = x_source_2[energy_cut], y_source_2[energy_cut], z_source_2[energy_cut], compton_crystal_2[energy_cut]

        true = (x_source_1_temp == x_source_2_temp) & (y_source_1_temp == y_source_2_temp) & (z_source_1_temp == z_source_2_temp) & (compton_crystal_1_temp == 1) & (compton_crystal_2_temp == 1)
        true2 = (compton_crystal_1_temp == 1) & (compton_crystal_2_temp == 1)

        if np.sum(energy_cut) > 0:
            a[ii] = np.sum(true) / np.sum(energy_cut)

        b[ii] = np.sum(true)
        c[ii] = np.sum(true2)

    # todo: Think about filtering options
    # todo: Scattering test
    # todo: Angle test
    # todo: Geometrical cut

    print(b[e_cut == 0.05] / b[e_cut == 0.2])
    print()

    fig, ax = plt.subplots()
    ax.plot(e_cut, a)
    y_lim = ax.get_ylim()
    ax.plot([0.05, 0.05], y_lim, linestyle='--', color='black')
    ax.plot([0.2, 0.2], y_lim, linestyle='--', color='black')
    ax.set_ylim(y_lim)
    ax_twin = ax.twinx()
    ax_twin.plot(e_cut, b)
    ax_twin.plot(e_cut, c)
    # ax_twin.set_yscale('log')
    plt.show()

    sys.exit()


    #
    energy_1 = coincidences_all['energy1']
    energy_2 = coincidences_all['energy2']
    e_cut = 0.25
    crit = (energy_1 >= e_cut) & (energy_2 >= e_cut)
    energies = np.concatenate((energy_1[crit], energy_2[crit]))

    energies_2 = np.concatenate((coincidences_cut['energy1'], coincidences_cut['energy2']))

    energy_spectrum([energies_2, energies],['Cut', 'All cut'])

    return 0


if __name__ == "__main__":
    main()
