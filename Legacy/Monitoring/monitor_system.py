"""
Monitor CPU and memory utilization

Author: Martin Rädler
"""
# Python libraries
import sys
from os import mkdir
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from datetime import datetime
from time import sleep
from psutil import cpu_count, cpu_percent, virtual_memory, process_iter, disk_partitions, disk_usage, disk_io_counters


def monitor_system():
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--sampling-frequency', type=float, help='Sampling frequency in Hz.', required=False, default=1.)
    parser.add_argument('--seconds-per-cycle', type=float, help='Duration of one cycle in seconds.', required=True)
    parser.add_argument('--number-of-cycles', type=int, help='Number of cycles.', required=True)
    args = parser.parse_args()

    print('Total tracking period: %d s' % (args.number_of_cycles * args.seconds_per_cycle))

    start_time = datetime.now()
    date_time_string = '%d-%d-%d_%d-%d-%d' % (start_time.year, start_time.month, start_time.day,
                                              start_time.hour, start_time.minute, start_time.second)
    output_dir = sys.path[0] + '/SYSTEM_MONITORING/' + date_time_string

    device_path = ''
    for part in disk_partitions():
        if part.mountpoint.startswith('/data/local1'):
            device_path = part.device
    device_name = device_path.removeprefix('/dev/')

    # print(disk_usage('/data/local1'))


    mkdir(output_dir)

    # n_cpu = cpu_count()
    cpu_utilization, memory_utilization, disk_utilization, time = [], [], [], []

    # cpu_percent(interval=None, percpu=True)

    disk_previous = disk_io_counters(perdisk=True)[device_name].write_bytes
    time_previous = datetime.now()

    jj = 1
    pbar = tqdm(total=args.number_of_cycles)
    while jj <= args.number_of_cycles:
        cpu_utilization.append(cpu_percent(interval=None, percpu=True))
        memory_utilization.append(virtual_memory().percent)
        time.append(datetime.now())

        elapsed = (datetime.now() - start_time).total_seconds()

        if elapsed > (jj * args.seconds_per_cycle):
            output_file = output_dir + '/system_utilization_%d.npz' % jj
            # print('Writing: %s' % output_file)
            np.savez(output_file,
                     cpu_utilization=np.array(cpu_utilization),
                     memory_utilization=np.array(memory_utilization),
                     time=np.array(time, dtype=object))
            pbar.update()

            del cpu_utilization, memory_utilization, disk_utilization, time
            cpu_utilization, memory_utilization, disk_utilization, time = [], [], [], []
            jj += 1

        sleep(1 / args.sampling_frequency)

    pbar.close()

    return 0


if __name__ == "__main__":
    monitor_system()
