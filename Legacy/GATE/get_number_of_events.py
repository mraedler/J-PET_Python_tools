"""
Get the total number of events

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import dirname
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    home_dir = dirname(dirname(sys.path[0]))
    geom_path = home_dir + '/J-PET/Gate_Output/Sensitivity_maps/TB_6_30_3_BI_4_18_3_Box'
    sim_paths = glob(geom_path + '/*')

    entry = '# NumberOfEvents = '

    n_events = []
    for ii in range(len(sim_paths)):
        file = open(sim_paths[ii] + '/simulation_statistics.txt')
        n_events_temp = np.array([int(line.strip()[len(entry):]) for line in tqdm(file) if line.startswith(entry)])
        n_events.append(n_events_temp)
        file.close()
    n_events = np.concatenate(n_events)

    n_events_tot = np.sum(n_events)
    # print(n_events_tot / 1e9)

    source_dim = 2 * np.array([215., 215., 175.])  # [mm]
    source_vol = np.prod(source_dim)  # [mm^3]
    density = n_events_tot / source_vol

    voxel_dim = np.array([5., 5., 5.])  # [mm]
    voxel_vol = np.prod(voxel_dim)  # [mm^3]
    print(density * voxel_vol)
    # sys.exit()

    duration = 1.  # [s]
    activity = 1e6  # [Bq]
    exp_mean = duration * activity
    exp_std = np.sqrt(exp_mean)

    bin_edges = exp_mean + np.linspace(-5., 5., 101) * exp_std
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    h, _ = np.histogram(n_events, bins=bin_edges, density=True)
    gaussian = 1 / np.sqrt(2 * np.pi * exp_std ** 2) * np.exp(-(bin_centers - exp_mean) ** 2 / (2 * exp_std ** 2))

    fig, ax = plt.subplots()
    ax.bar(bin_centers, h, width=bin_width, alpha=0.5)
    ax.plot(bin_centers, gaussian, linewidth=2)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()



    return 0




if __name__ == "__main__":
    main()
