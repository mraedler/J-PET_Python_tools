"""
Analyze the sensitivity simulation

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from uproot import open

# Auxiliary functions
from sensitivity_coincidences import get_sensitivity, plot_sensitivity


def main():
    root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-04-19_17-56-57/results.root'
    root_file = open(root_path)
    coincidences = root_file['MergedCoincidences']
    # print(coincidences.keys())

    x1, y1, z1 = np.array(coincidences['sourcePosX1']), np.array(coincidences['sourcePosY1']), np.array(coincidences['sourcePosZ1'])
    # x2, y2, z2 = np.array(coincidences['sourcePosX2']), np.array(coincidences['sourcePosY2']), np.array(coincidences['sourcePosZ2'])
    t1 = np.array(coincidences['time1'])
    # t2 = np.array(coincidences['time2'])

    n_bins = 80
    z_edges = np.linspace(-1200., 1200., n_bins + 1)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_widths = z_edges[1:] - z_edges[:-1]
    h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain = get_sensitivity(root_path, z_edges)
    plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate,
                     h_filtered_brain, 1, vertical_lines=[-815. - 330. / 2, -815. + 330. / 2])

    print(t1[0])
    print(t1[-1])

    print(np.sum(z1 <= -650))
    print(np.sum(z1 > -650))
    print(z1.size)

    t_edges = np.logspace(-9, 1, 200)
    t_centers = (t_edges[1:] + t_edges[:-1]) / 2
    t_width = t_edges[1:] - t_edges[:-1]
    h, _ = np.histogram(np.diff(t1), bins=t_edges)
    # h, _ = np.histogram(t1, bins=t_edges)
    fig, ax = plt.subplots()
    ax.bar(t_centers, h, width=t_width)
    ax.set_xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(z1, x1)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
