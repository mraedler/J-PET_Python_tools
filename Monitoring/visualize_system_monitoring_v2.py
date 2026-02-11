"""
Visualize the system monitoring

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import basename
from datetime import date
from datetime import datetime
from glob import glob
from natsort import natsorted
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter


def visualize_system_monitoring():
    # Load
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-10-7_15-11-13/'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-10-7_15-27-55/'
    # input_dir = sys.path[0] + '/SYSTEM_MONITORING/2025-10-10_17-44-15/'
    # input_dir = sys.path[0] + '/PROCESS_MONITORING/2025-10-10_17-44-29/'
    input_dir = sys.path[0] + '/Process_data/2025-10-23_15-26-34/'

    time = np.fromfile(input_dir + 'time.bin', dtype=np.float64)
    # cpu_temp = np.fromfile(input_dir + 'cpu_temp.bin', dtype=np.float32)
    cpu_temp = np.zeros(time.size)
    cpu_use, n_cpu = load_binary_file(input_dir, 'cpu_use_', np.float32)
    memory_use, total_memory = load_binary_file(input_dir, 'memory_use_', np.float32)
    disk_write, disk_name = load_binary_file(input_dir, 'disk_write_', np.int64)
    disk_read, disk_name = load_binary_file(input_dir, 'disk_read_', np.int64)

    #
    time = np.array([datetime.fromtimestamp(time[ii]) for ii in range(time.size)])
    # cpu_use = np.reshape(cpu_use, (int(cpu_use.size / int(n_cpu)), int(n_cpu)), order='C')

    print(time[-1] - time[0])

    delta_time = np.diff(time)
    time_between = time[:-1] + delta_time / 2
    delta_time = np.array([d.total_seconds() for d in delta_time])
    disk_write_rate = np.diff(disk_write) / delta_time
    disk_read_rate = np.diff(disk_read) / delta_time
    print(disk_write[-1] - disk_write[0])


    plt.rcParams.update({'font.size': 16})
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    ax0.plot(time, cpu_use, alpha=0.5)
    # ax0.plot(time, np.mean(cpu_use, axis=1), alpha=0.5)
    # ax0.plot(time, moving_average(cpu_use, 60), label='CPU use', color='tab:blue')
    # ax0.plot(time, moving_average(np.mean(cpu_use, axis=1), 60), label='CPU use', color='tab:blue')
    ax0.set_xticks(ax0.get_xticks()[::4])
    ax0.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
    ax0.set_ylabel('Percentage [%]')
    ax0.set_title('CPU usage')

    ax1.plot(time, memory_use, alpha=0.5)
    ax1.plot(time, moving_average(memory_use, 60), color='tab:blue')
    ax1.set_xticks(ax1.get_xticks()[::4])
    ax1.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
    ax1.set_ylabel('Percentage [%]')
    ax1.set_title('Memory usage')

    ax2.plot(time, cpu_temp, alpha=0.5)
    ax2.plot(time, moving_average(cpu_temp, 60), color='tab:blue')
    ax2.set_xticks(ax2.get_xticks()[::4])
    ax2.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
    ax2.set_ylabel('Degree Celsius [°C]')
    ax2.set_title('CPU temperature')

    # ax3.plot(time, disk_write / 1e9)
    # ax3.plot(time_between, disk_write_rate / 1e9, alpha=0.5)
    # ax3.plot(time_between, moving_average(disk_write_rate, 60) / 1e9, color='tab:blue')
    # ax3.plot(time_between, disk_read_rate / 1e9, alpha=0.5, color='tab:orange')
    # ax3.plot(time_between, moving_average(disk_read_rate, 60) / 1e9, color='tab:orange')
    ax3.plot(time, disk_write)
    ax3.set_xticks(ax3.get_xticks()[::4])
    ax3.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
    ax3.set_ylabel('Byte rate [GB/s]')
    ax3.set_title('Disk use')

    plt.show()

    return 0


def load_binary_file(input_dir, prefix, dtype):
    file_path = glob(input_dir + prefix + '*.bin')[0]
    file_name = basename(file_path)
    file_name_info = file_name[len(prefix):-4]  # ".bin" has four characters
    data = np.fromfile(file_path, dtype=dtype)

    return data, file_name_info


def moving_average(y, window_half_size):
    window = np.ones(2 * window_half_size + 1)
    y_moving_sum = np.convolve(y, window, mode='full')[window_half_size:-window_half_size]
    normalization = np.convolve(np.ones(y.size), window, mode='full')[window_half_size:-window_half_size]
    y_moving_average = y_moving_sum / normalization
    return y_moving_average


if __name__ == "__main__":
    visualize_system_monitoring()
