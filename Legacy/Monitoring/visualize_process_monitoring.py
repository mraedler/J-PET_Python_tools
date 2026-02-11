"""
Visualize the process monitoring

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from glob import glob
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter


def main():
    input_dirs = [sys.path[0] + '/PROCESS_MONITORING/2025-6-26_17-36-30/*',
                  sys.path[0] + '/PROCESS_MONITORING/2025-6-27_12-22-36/*']

    time_list, memory_utilization_list = load_monitoring_data(input_dirs)

    labels = ['200 keV', '100 keV']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax_in = fig.add_axes([0.25, 0.25, 0.5, 0.5])
    for ii in range(len(input_dirs)):
        time = date2num(time_list[ii]) - date2num(time_list[ii][0])
        ax.plot(time, memory_utilization_list[ii])
        ax_in.plot(time, memory_utilization_list[ii], label=labels[ii])

    time = date2num(time_list[0]) - date2num(time_list[0][0])
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks(np.linspace(time[0], time[-1], 6))
    # ax.xaxis.set_major_formatter(DateFormatter("%M:%S"))
    # ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.xaxis.set_major_formatter(DateFormatter("%H"))
    # ax.set_xlabel('Time [hours:minutes]')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('RSS [GB]')

    ax_in.set_ylim(54.5, 57.5)
    ax_in.set_xlim(time[0], time[-1])
    ax_in.set_xticks(np.linspace(time[0], time[-1], 3))
    # ax_in.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax_in.xaxis.set_major_formatter(DateFormatter("%H"))
    ax_in.legend(loc='lower center', title='Energy threshold', frameon=False)

    plt.show()

    return 0


def load_monitoring_data(input_dir_list):

    time_list, memory_utilization_list = [], []

    for ii in range(len(input_dir_list)):

        paths = natsorted(glob(input_dir_list[ii]))
        time, memory_utilization = [], []

        for jj in range(len(paths)):
            numpy_file = np.load(paths[jj], allow_pickle=True)

            time.append(numpy_file['time'])
            memory_utilization.append(numpy_file['memory_utilization'])

        time_list.append(np.concatenate(time))
        memory_utilization_list.append(np.concatenate(memory_utilization))

    return time_list, memory_utilization_list


if __name__ == "__main__":
    main()
