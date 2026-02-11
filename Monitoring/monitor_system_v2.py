"""
Monitor CPU, memory, and disk use (optional)

Author: Martin Rädler
"""
# Python libraries
import sys
from struct import pack
from os import mkdir
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime
from time import sleep
from psutil import cpu_count, cpu_percent, virtual_memory, disk_partitions, disk_usage, disk_io_counters, sensors_temperatures


def monitor_system():
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--sampling-frequency', type=float, help='Sampling frequency in Hz.', required=False, default=1.)
    parser.add_argument('--hours', type=float, help='Monitoring duration in hours.', required=False, default=0.)
    parser.add_argument('--minutes', type=float, help='Monitoring duration in minutes.', required=False, default=0.)
    parser.add_argument('--mountpoint', type=str, help='Mount point of the disk.', required=False, default=None)
    args = parser.parse_args()

    print('Monitoring the system for %d hours and %d minutes.' % (args.hours, args.minutes))
    monitoring_duration_seconds = args.hours * 60 * 60 + args.minutes * 60

    # Output directory
    start_time = datetime.now()
    date_time_string = '%d-%d-%d_%d-%d-%d' % (start_time.year, start_time.month, start_time.day,
                                              start_time.hour, start_time.minute, start_time.second)
    output_dir = sys.path[0] + '/SYSTEM_MONITORING/' + date_time_string

    mkdir(output_dir)

    n_cpu = cpu_count()
    time = open(output_dir + '/time.bin', 'ab')
    cpu_temp = open(output_dir + '/cpu_temp.bin', 'ab')
    cpu_use = open(output_dir + '/cpu_use_%d.bin' % n_cpu, 'ab')
    memory_use = open(output_dir + '/memory_use_%d_GB.bin' % (virtual_memory().total / 1e9), 'ab')

    device_name, disk_write, disk_read = None, None, None
    if args.mountpoint is not None:
        counter = 0
        for part in disk_partitions():
            if part.mountpoint == args.mountpoint:
                device_name = part.device.removeprefix('/dev/')
                counter += 1
        if counter != 1:
            print('Error: %d devices found for the provided mount point: %s. Deactivating disk monitoring.' % (counter, args.mountpoint))
            args.mountpoint = None
        else:
            # print(disk_usage(args.mountpoint))
            disk_write = open(output_dir + '/disk_write_%s.bin' % device_name, 'ab')
            disk_read = open(output_dir + '/disk_read_%s.bin' % device_name, 'ab')

    #
    elapsed_time = (datetime.now() - start_time).total_seconds()
    pbar = tqdm(total=int(monitoring_duration_seconds))
    cpu_fmt = f'{n_cpu}f'

    while elapsed_time < monitoring_duration_seconds:
        time_before = datetime.now()
        time.write(pack('d', time_before.timestamp()))
        cpu_temp.write(pack('f', sensors_temperatures(fahrenheit=False)['k10temp'][0].current))
        cpu_use.write(pack(cpu_fmt, *cpu_percent(interval=None, percpu=True)))
        memory_use.write(pack('f', virtual_memory().percent))

        if args.mountpoint is not None:
            io = disk_io_counters(perdisk=True)[device_name]
            disk_write.write(pack('Q', io.write_bytes))
            disk_read.write(pack('Q', io.read_bytes))

        # # Write after every iteration
        # time.flush()
        # cpu_use.flush()
        # memory_use.flush()
        # disk_write.flush()
        # disk_read.flush()

        elapsed_time = (datetime.now() - start_time).total_seconds()
        pbar.update(int(elapsed_time) - pbar.n)
        time_diff = (datetime.now() - time_before).total_seconds()
        sleep(1 / args.sampling_frequency - time_diff)

    time.close()
    cpu_temp.close()
    cpu_use.close()
    memory_use.close()
    if args.mountpoint is not None:
        disk_write.close()
        disk_read.close()

    pbar.close()

    return 0


if __name__ == "__main__":
    monitor_system()
