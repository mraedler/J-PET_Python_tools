"""
Analyze the export rate of GATE

@author: Martin Rädler
"""
# Python libraries
import sys
import datetime

import numpy
import numpy as np
from pandas import read_csv, to_datetime
from matplotlib.dates import date2num, DateFormatter
import matplotlib.pyplot as plt


def main():
    export_tracking_path = '/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/ExportTracking'
    # df = read_csv(export_tracking_path + '/2025-10-01_17-35-15.log', sep=r"\s+", header=None)
    # df = read_csv(export_tracking_path + '/2025-10-02_15-28-25.log', sep=r"\s+", header=None)
    # df = read_csv(export_tracking_path + '/2025-10-03_15-17-30.log', sep=r"\s+", header=None)
    # df = read_csv(export_tracking_path + '/2025-10-07_15-28-34.log', sep=r"\s+", header=None)
    # df = read_csv(export_tracking_path + '/2025-10-10_17-44-58.log', sep=r"\s+", header=None)
    df = read_csv(export_tracking_path + '/2025-10-23_15-26-58.log', sep=r"\s+", header=None)

    # time = to_datetime(df.iloc[:, 0], format='%Y-%m-%d_%H-%M-%S').to_numpy()
    # time = to_datetime(df.iloc[:, 0], format='%Y-%m-%d_%H-%M-%S').dt.to_pydatetime()
    time_before = to_datetime(df.iloc[:, 0], unit='s', utc=True).dt.tz_convert('Europe/Berlin').dt.to_pydatetime()
    time_after = to_datetime(df.iloc[:, -1], unit='s', utc=True).dt.tz_convert('Europe/Berlin').dt.to_pydatetime()
    time = time_before + (time_after - time_before) / 2

    # Load the memory data and remove NAN entries
    memory = df.iloc[:, 1:-1].to_numpy()
    first_column = memory[:, 0]
    idx = 0
    for ii in range(first_column.size):
        if first_column[ii] != 'NAN':
            idx = ii
            break
    memory = memory[idx:, :].astype(np.int64)
    time = time[idx:]

    #
    mean_memory = np.mean(memory, axis=1)
    time_delta = np.diff(time)
    time_mid = time[:-1] + time_delta / 2
    time_delta_seconds = np.array([td.total_seconds() for td in time_delta])
    d_memory = np.diff(mean_memory) / time_delta_seconds

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, memory / 1e9, color='tab:blue', alpha=0.01)
    ax.plot(time, mean_memory / 1e9, linewidth=2, color='tab:blue')
    ax.set_xticks(ax.get_xticks()[1::4])
    ax.xaxis.set_major_formatter(DateFormatter("%d.%m %H:%M"))
    ax.set_xlabel('Time [day.month hour:minute]')
    ax.set_ylabel('Output root file size [GB]', color='tab:blue')
    ax.spines['left'].set_color('tab:blue')
    ax.tick_params(axis='y', colors='tab:blue')

    ax_twin = ax.twinx()
    ax_twin.plot(time_mid, d_memory / 1e3, color='tab:orange', alpha=0.1)
    ax_twin.plot(time_mid, moving_average(d_memory, 60) / 1e3, color='tab:orange')
    # ax_twin.plot(time_mid, moving_average(d_memory, 500) / 1e3, color='tab:orange')
    ax_twin.set_ylabel('Byte rate [kB/s]', color='tab:orange')
    ax_twin.spines['right'].set_color('tab:orange')
    ax_twin.tick_params(axis='y', colors='tab:orange')
    plt.show()


    return 0


def moving_average(y, window_half_size):
    window = np.ones(2 * window_half_size + 1)
    y_moving_sum = np.convolve(y, window, mode='full')[window_half_size:-window_half_size]
    normalization = np.convolve(np.ones(y.size), window, mode='full')[window_half_size:-window_half_size]
    y_moving_average = y_moving_sum / normalization
    return y_moving_average


if __name__ == "__main__":
    main()
