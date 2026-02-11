"""
Visualize the system monitoring

Author: Martin Rädler
"""
# Python libraries
import sys
from datetime import date
from glob import glob
from natsort import natsorted
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter


def visualize_system_monitoring():
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-1-31_17-36-14/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-3_13-22-45/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-3_17-41-16/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-4_18-17-22/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-5_18-26-30/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-6_18-31-59/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-7_17-17-24/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-13_19-4-24/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-18_14-40-36/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-2-27_17-56-52/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-3-19_15-20-55/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-3-25_14-18-35/*'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-4-8_14-40-13/*'
    # input_dirs = [sys.path[0] + '/SYSTEM_MONITORING/2025-6-23_16-37-37/*',
    #               sys.path[0] + '/SYSTEM_MONITORING/2025-6-23_17-8-42/*',
    #               sys.path[0] + '/SYSTEM_MONITORING/2025-6-24_8-35-17/*']
    # input_dirs = [sys.path[0] + '/SYSTEM_MONITORING/2025-6-24_12-31-2/*']
    # input_dirs = [sys.path[0] + '/SYSTEM_MONITORING/2025-10-1_17-34-7/*']
    # input_dirs = [sys.path[0] + '/SYSTEM_MONITORING/2025-10-2_15-27-29/*']
    input_dirs = [sys.path[0] + '/SYSTEM_MONITORING/2025-10-3_15-17-38/*']

    cpu_utilization_list, memory_utilization_list, time_list = load_monitoring_data(input_dirs)


    # labels = ['v9.4 (subdivided)', 'v9.4 with mem. leak fix (subdivided)', 'v9.4 with mem. leak fix (uninterruped)']
    # labels = ['v9.4 with mem. leak fix (uninterruped)']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    for ii in range(len(input_dirs)):
        # time = date2num(time_list[ii]) - date2num(time_list[ii][0])
        time = date2num(time_list[ii])

        mean_cup_utilization = np.mean(cpu_utilization_list[ii], axis=1)
        bb = moving_average(mean_cup_utilization, window_half_size=120)


        # ax.plot(time, cpu_utilization_list[ii])
        ax.plot(time, mean_cup_utilization, label='Average CPU utilization')
        ax.plot(time, bb, label='Moving average CPU utilization')
        ax.plot(time, memory_utilization_list[ii], label='Memory utilization')



    # time = date2num(time_list[0]) - date2num(time_list[0][0])
    # ax.set_xlim(time[0], time[-1])
    # ax.set_xticks(np.linspace(time[0], time[-1], 6))
    ax.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
    # ax.xaxis.set_major_formatter(DateFormatter("%M:%S"))
    # ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    # ax.set_yticks(np.arange(0, 25, 5))
    # ax.set_yticks(np.arange(0, 12, 2))
    # ax.set_xlabel('Time [min:seconds]')
    ax.set_xlabel('Time [hours:minutes]')
    ax.set_ylabel('Percentage [%]')
    ax.legend(loc='upper right', frameon=False)
    plt.show()

    return 0


def load_monitoring_data(input_dir_list):

    cpu_utilization_list, memory_utilization_list, time_list = [], [], []

    for ii in range(len(input_dir_list)):

        paths = natsorted(glob(input_dir_list[ii]))
        cpu_utilization, memory_utilization, time = [], [], []

        for jj in range(len(paths)):
            numpy_file = np.load(paths[jj], allow_pickle=True)

            cpu_utilization.append(numpy_file['cpu_utilization'])
            memory_utilization.append(numpy_file['memory_utilization'])
            time.append(numpy_file['time'])

        cpu_utilization_list.append(np.concatenate(cpu_utilization))
        memory_utilization_list.append(np.concatenate(memory_utilization))
        time_list.append(np.concatenate(time))

    return cpu_utilization_list, memory_utilization_list, time_list


def moving_average(y, window_half_size):
    window = np.ones(2 * window_half_size + 1)
    y_moving_sum = np.convolve(y, window, mode='full')[window_half_size:-window_half_size]
    normalization = np.convolve(np.ones(y.size), window, mode='full')[window_half_size:-window_half_size]
    y_moving_average = y_moving_sum / normalization
    return y_moving_average


if __name__ == "__main__":
    visualize_system_monitoring()
