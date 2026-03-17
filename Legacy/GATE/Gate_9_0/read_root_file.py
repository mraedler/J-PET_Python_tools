"""
Read root files

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from uproot import open
import matplotlib.pyplot as plt


def main():
    # root_file = open('/home/martin/J-PET/TB_J-PET/output/output.root')
    # root_file = open('/home/martin/J-PET/TB_J-PET_Edit/output/results.root')
    root_file = open('/home/martin/J-PET/TB_J-PET_Brain/output/results.root')
    # print(root_file.keys())
    hits = root_file['Hits']
    singles = root_file['Singles']
    singles_no_wls = root_file['SinglesNoWLS']
    coincidences = root_file['Coincidences']
    # latest_event_id = root_file['latest_event_ID']  # Histogram
    # total_nb_primaries = root_file['total_nb_primaries']  # Histogram
    pet_data = root_file['pet_data']
    optical_data = root_file['OpticalData']

    # Hits
    # print(hits.keys())
    hits_pos_x = np.array(hits['posX'])
    hits_pos_y = np.array(hits['posY'])
    hits_pos_z = np.array(hits['posZ'])

    hits_source_pos_x = np.array(hits['sourcePosX'])
    hits_source_pos_y = np.array(hits['sourcePosY'])
    hits_source_pos_z = np.array(hits['sourcePosZ'])

    # Singles (no WLS)
    # print(singles_no_wls.keys())
    singles_global_pos_x = np.array(singles_no_wls['globalPosX'])
    singles_global_pos_y = np.array(singles_no_wls['globalPosY'])
    singles_global_pos_z = np.array(singles_no_wls['globalPosZ'])

    singles_source_pos_x = np.array(singles_no_wls['sourcePosX'])
    singles_source_pos_y = np.array(singles_no_wls['sourcePosY'])
    singles_source_pos_z = np.array(singles_no_wls['sourcePosZ'])

    # Coincidences
    print(coincidences.keys())
    coincidences_source_pos_x_1 = np.array(coincidences['sourcePosX1'])
    coincidences_source_pos_y_1 = np.array(coincidences['sourcePosY1'])
    coincidences_source_pos_z_1 = np.array(coincidences['sourcePosZ1'])

    coincidences_source_pos_x_2 = np.array(coincidences['sourcePosX2'])
    coincidences_source_pos_y_2 = np.array(coincidences['sourcePosY2'])
    coincidences_source_pos_z_2 = np.array(coincidences['sourcePosZ2'])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(source_pos_x, source_pos_y, source_pos_z, s=.1)
    # ax.scatter(pos_x, pos_y, pos_z, s=.1)
    # plt.show()
    # sys.exit()

    # # Hits vs. Singles
    # fig, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.scatter(hits_pos_x, hits_pos_y, s=.1)
    # ax0.scatter(hits_source_pos_x, hits_source_pos_y, s=.1)
    # ax0.set_aspect(1)
    # ax0.set_title('Hits')
    #
    # ax1.scatter(singles_global_pos_x, singles_global_pos_y, s=.1)
    # ax1.scatter(singles_source_pos_x, singles_source_pos_y, s=.1)
    # ax1.set_aspect(1)
    # ax1.set_title('Singles (no WLS)')
    # plt.show()

    # Analyze absorption
    hits_source_pos_rho = np.sqrt(hits_source_pos_x ** 2 + hits_source_pos_y ** 2)
    singles_source_pos_rho = np.sqrt(singles_source_pos_x ** 2 + singles_source_pos_y ** 2)
    coincidences_source_pos_rho_1 = np.sqrt(coincidences_source_pos_x_1 ** 2 + coincidences_source_pos_y_1 ** 2)

    bin_edges = np.linspace(0., 101., 102)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    h_hits, _ = np.histogram(hits_source_pos_rho, bins=bin_edges)
    h_singles, _ = np.histogram(singles_source_pos_rho, bins=bin_edges)
    h_coincidences, _ = np.histogram(coincidences_source_pos_rho_1, bins=bin_edges)

    fig, ax = plt.subplots()
    ax.bar(bin_centers, h_hits / bin_centers, width=bin_width)
    ax.bar(bin_centers, h_singles / bin_centers, width=bin_width)
    ax.bar(bin_centers, h_coincidences / bin_centers, width=bin_width)
    plt.show()

    return 0


if __name__ == '__main__':
    main()
