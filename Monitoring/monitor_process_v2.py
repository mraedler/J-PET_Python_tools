"""
Monitor memory utilization of a specific process

Author: Martin Rädler
"""
# Python libraries
import sys
from os import mkdir
from struct import pack
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from time import sleep
from tqdm import trange
from psutil import cpu_count, virtual_memory, process_iter, AccessDenied, NoSuchProcess


def monitor_process():
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--process', type=str, help='Process name to be tracked', required=True)
    parser.add_argument('--sampling-frequency', type=float, help='Sampling frequency in Hz.', required=False, default=1.)
    parser.add_argument('--hours', type=float, help='Monitoring duration in hours.', required=False, default=0.)
    parser.add_argument('--minutes', type=float, help='Monitoring duration in minutes.', required=False, default=0.)
    args = parser.parse_args()

    print('Monitoring the system for %d hours and %d minutes.' % (args.hours, args.minutes))
    monitoring_duration_seconds = args.hours * 60 * 60 + args.minutes * 60

    # Output directory
    start_time = datetime.now()
    date_time_string = '%d-%d-%d_%d-%d-%d' % (start_time.year, start_time.month, start_time.day,
                                              start_time.hour, start_time.minute, start_time.second)
    output_dir = sys.path[0] + '/Process_data/' + date_time_string
    mkdir(output_dir)

    n_cpu = cpu_count()
    time = open(output_dir + '/time.bin', 'ab')
    memory_use = open(output_dir + '/memory_use_%d_GB.bin' % (virtual_memory().total / 1e9), 'ab')
    cpu_use = open(output_dir + '/cpu_use_%d.bin' % n_cpu, 'ab')
    disk_write = open(output_dir + '/disk_write_%s.bin' % args.process, 'ab')
    disk_read = open(output_dir + '/disk_read_%s.bin' % args.process, 'ab')

    # Start the monitoring
    n_steps = int(round(monitoring_duration_seconds * args.sampling_frequency))
    for ii in trange(n_steps):
        # process_counter = 0
        cpu_use_tot = 0
        memory_use_tot = 0
        disk_write_tot = 0
        disk_read_tot = 0

        for proc in process_iter(['name']):
            if proc.info['name'] == args.process:
                # process_counter += 1
                cpu_use_tot += proc.cpu_percent()
                memory_use_tot += proc.memory_info().rss
                # memory_use_tot += proc.memory_info().vms
                try:
                    io = proc.io_counters()
                    disk_write_tot += io.write_bytes
                    disk_read_tot += io.read_bytes
                except (NoSuchProcess, AccessDenied):
                    pass

        # Write
        time.write(pack('d', datetime.now().timestamp()))
        # cpu_use.write(pack('f', cpu_use_tot / process_counter))
        cpu_use.write(pack('f', cpu_use_tot / n_cpu))
        memory_use.write(pack('f', memory_use_tot / 1e9))  # GB
        disk_write.write(pack('Q', disk_write_tot))
        disk_read.write(pack('Q', disk_read_tot))

        elapsed_time = (datetime.now() - start_time).total_seconds()
        next_time = (ii + 1) / args.sampling_frequency
        sleep_time = next_time - elapsed_time

        if sleep_time > 0.:
            sleep(sleep_time)

    # Close the output files
    time.close()
    cpu_use.close()
    memory_use.close()
    disk_write.close()
    disk_read.close()

    return 0


if __name__ == '__main__':
    monitor_process()
