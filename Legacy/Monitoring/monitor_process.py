"""
Monitor CPU and memory utilization of a specific process

Author: Martin Rädler
"""
# Python libraries
import sys
from os import mkdir
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from time import sleep
from tqdm import tqdm
from psutil import process_iter


def monitor_process():
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
    output_dir = sys.path[0] + '/PROCESS_MONITORING/' + date_time_string
    print(output_dir)
    # mkdir(output_dir)

    time, memory_utilization = [], []

    jj = 1
    pbar = tqdm(total=args.number_of_cycles)
    while jj <= args.number_of_cycles:

        memory_utilization_temp = 0
        # process_counter = 0
        for proc in process_iter(['name']):
            # if proc.info['name'] == 'Gate':
            if proc.info['name'] == 'chrome':
                # print(proc.memory_info())
                # print(proc.cpu_affinity())
                # print(proc.cpu_num())
                print(proc.cpu_percent())
                # print(proc.cpu_times())
                # print(dir(proc))
                print(proc.io_counters())
                memory_utilization_temp += proc.memory_info().rss / (1024 ** 3)  # GB
                # memory_utilization_temp += proc.memory_info().vms / (1024 ** 3)  # GB
                # process_counter += 1
        print()
        time.append(datetime.now())
        memory_utilization.append(memory_utilization_temp)

        elapsed = (datetime.now() - start_time).total_seconds()

        if elapsed > (jj * args.seconds_per_cycle):
            output_file = output_dir + '/process_utilization_%d.npz' % jj
            # print('Writing: %s' % output_file)
            np.savez(output_file,
                     memory_utilization=np.array(memory_utilization),
                     time=np.array(time, dtype=object))
            pbar.update()

            del time, memory_utilization
            time, memory_utilization = [], []
            jj += 1

        sleep(1 / args.sampling_frequency)

    return 0


if __name__ == '__main__':
    monitor_process()
