"""
Convert data structures

@author: Martin Rädler
"""
# Python libraries
import sys
from os import mkdir
from os.path import split, isdir, isfile, dirname, abspath
import numpy as np
from tqdm import tqdm


def load_or_convert_to_structured_array(ttree, keys=None, overwrite=False):
    # Get and split the path of the original root file
    root_file_path = ttree.file.file_path
    root_file_path_split = split_path(root_file_path)

    # Path, where the structured arrays are stored
    # struct_arr_dir = sys.path[0] + '/Structured_arrays/' + root_file_path_split[-2] + '/'
    # struct_arr_dir = '/home/martin/PycharmProjects/J-PET/Structured_arrays/' + root_file_path_split[-2] + '/'
    script_dir = dirname(abspath(__file__))
    struct_arr_dir = script_dir + '/Structured_arrays/' + root_file_path_split[-2] + '/'

    struct_arr_path = struct_arr_dir + ttree.name + root_file_path_split[-1][7:-5] + '.npy'

    #
    if keys is None:
        keys = ttree.keys()
        print(keys)
        sys.exit()

    if isfile(struct_arr_path) and not overwrite:
        struct_arr = np.load(struct_arr_path, allow_pickle=True)

    else:
        # Create the directory, if it does not exist
        if not isdir(struct_arr_dir):
            mkdir(struct_arr_dir)

        # Create the structured array and save
        struct_arr = convert_gate_ttree_to_structured_array(ttree, keys)
        np.save(struct_arr_path, struct_arr)

    return struct_arr


def convert_gate_ttree_to_structured_array(gate_ttree, keys):
    # Construct the data type
    d_type = []
    for key, data_type in gate_ttree.itertypenames():
        if key in keys:
            d_type.append((key, adjust_data_type(data_type)))

    # Allocate the structured array
    struct_arr = np.empty(gate_ttree.num_entries, dtype=d_type)

    # Write into the structured array
    for key, branch in tqdm(gate_ttree.iteritems()):
        if key in keys:
            struct_arr[key] = np.array(branch)
            # struct_arr[key] = branch.array(library='np')

    return struct_arr


def adjust_data_type(data_type):
    if data_type == 'int32_t':
        data_type_adjusted = np.int32
    elif data_type == 'float':
        data_type_adjusted = np.float32
    elif data_type == 'double':
        data_type_adjusted = np.float64
    elif data_type == 'char*':
        data_type_adjusted = np.object_
        # data_type_adjusted = object
    else:
        sys.exit('Error: %s not known!' % data_type)

    return data_type_adjusted


def split_path(path):
    # Split the path into directory and filename
    head, tail = split(path)

    # Allocate list
    directories = []
    while head != '/':
        head, directory = split(head)
        directories.append(directory)

    # Reverse the list to get the correct order
    directories.reverse()

    # Add the filename to the list
    directories.append(tail)

    return directories
