"""
Compare the hits

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from uproot import open
import matplotlib.pyplot as plt


def main():
    """

    :return:
    """
    """Gate 9.0"""
    root_file = open('/home/martin/J-PET/Gate_mac_9.0/TB_J-PET_Brain_2/Output/2024-02-14_17-38-18/results.root')
    hits = root_file['Hits']
    crystal_compton_0 = np.array(hits['nCrystalCompton'])
    pos_x_0 = np.array(hits['posX'])
    pos_y_0 = np.array(hits['posY'])

    """Gate 9.3"""
    root_file = open('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-01-29_16-27-06/results.root')
    b_hits_1 = root_file['Hits_brainLayer_1']
    b_hits_2 = root_file['Hits_brainLayer_2']
    b_hits_3 = root_file['Hits_brainLayer_3']
    tb_hits_1 = root_file['Hits_layer_1']
    tb_hits_2 = root_file['Hits_layer_2']
    tb_hits_3 = root_file['Hits_layer_3']

    crystal_compton_3 = np.concatenate((np.array(b_hits_1['nCrystalCompton']), np.array(b_hits_2['nCrystalCompton']),
                                        np.array(b_hits_3['nCrystalCompton']), np.array(tb_hits_1['nCrystalCompton']),
                                        np.array(tb_hits_2['nCrystalCompton']), np.array(tb_hits_3['nCrystalCompton'])))

    pos_x_3 = np.concatenate((np.array(b_hits_1['posX']), np.array(b_hits_2['posX']), np.array(b_hits_3['posX']),
                              np.array(tb_hits_1['posX']), np.array(tb_hits_2['posX']), np.array(tb_hits_3['posX'])))

    pos_y_3 = np.concatenate((np.array(b_hits_1['posY']), np.array(b_hits_2['posY']), np.array(b_hits_3['posY']),
                              np.array(tb_hits_1['posY']), np.array(tb_hits_2['posY']), np.array(tb_hits_3['posY'])))

    """abc"""
    b_0 = np.bincount(crystal_compton_0)
    b_3 = np.bincount(crystal_compton_3)

    # fig, ax = plt.subplots()
    # ax.bar(np.arange(b_0.size), b_0, alpha=0.5)
    # ax.bar(np.arange(b_3.size), b_3, alpha=0.5)
    # ax.set_yscale('log')
    # plt.show()

    """"""

    rho_0 = np.sqrt(pos_x_0 ** 2 + pos_y_0 ** 2)
    rho_3 = np.sqrt(pos_x_3 ** 2 + pos_y_3 ** 2)

    rand_idx_0 = np.random.choice(rho_0.size, int(0.01 * rho_0.size), replace=False)
    rand_idx_3 = np.random.choice(rho_3.size, int(0.01 * rho_3.size), replace=False)

    rho_edges = np.linspace(0, 500, 100)
    rho_centers = (rho_edges[1:] + rho_edges[:-1]) / 2
    rho_widths = rho_edges[1:] - rho_edges[:-1]



    fig, ax = plt.subplots()

    for ii in range(10):
        h_temp, _ = np.histogram(rho_0[crystal_compton_0 == ii], bins=rho_edges)
        # h_temp, _ = np.histogram(rho_3[crystal_compton_3 == ii], bins=rho_edges)
        ax.bar(rho_centers, h_temp, width=rho_widths, alpha=0.5, label=str(ii))

    # ax.scatter(rho_0[rand_idx_0], crystal_compton_0[rand_idx_0], alpha=0.1)
    # ax.scatter(rho_0[rand_idx_0], crystal_compton_0[rand_idx_0], alpha=0.1)
    ax.set_ylim(0, 1e6)
    ax.legend()
    plt.show()



    return 0


def hit_positions_9_3(root_file):
    b_hits_1 = root_file['Hits_brainLayer_1']
    b_hits_2 = root_file['Hits_brainLayer_2']
    b_hits_3 = root_file['Hits_brainLayer_3']

    # tb_hits_1 = root_file['Hits_layer_1']
    # tb_hits_2 = root_file['Hits_layer_2']
    # tb_hits_3 = root_file['Hits_layer_3']

    b_hits_1_pdg = np.array(b_hits_1['PDGEncoding'])
    b_hits_2_pdg = np.array(b_hits_2['PDGEncoding'])
    b_hits_3_pdg = np.array(b_hits_3['PDGEncoding'])

    b_hits_1_pos_x = np.array(b_hits_1['posX'])[b_hits_1_pdg == 22]
    b_hits_1_pos_y = np.array(b_hits_1['posY'])[b_hits_1_pdg == 22]
    b_hits_2_pos_x = np.array(b_hits_2['posX'])[b_hits_2_pdg == 22]
    b_hits_2_pos_y = np.array(b_hits_2['posY'])[b_hits_2_pdg == 22]
    b_hits_3_pos_x = np.array(b_hits_3['posX'])[b_hits_3_pdg == 22]
    b_hits_3_pos_y = np.array(b_hits_3['posY'])[b_hits_3_pdg == 22]

    fig, ax = plt.subplots()
    ax.scatter(b_hits_1_pos_x, b_hits_1_pos_y)
    ax.scatter(b_hits_2_pos_x, b_hits_2_pos_y)
    ax.scatter(b_hits_3_pos_x, b_hits_3_pos_y)
    plt.show()
    return 0


if __name__ == "__main__":
    main()
