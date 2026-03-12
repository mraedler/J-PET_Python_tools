"""
CASToR crystalID

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from re import split


def main():
    geom_file_path = '/home/martin/J-PET/CASToR/castor_v3.1.1/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_Brain.geom'
    read_geom_file(geom_file_path)

    return 0


def read_geom_file(geom_file_path):
    # Get the number of layers
    out = search_txt_file(geom_file_path, ['number of elements', 'number of layers'], 1)
    number_of_layers = int(out[1][0])

    #
    tags_list = ['number of rsectors', 'number of rsectors axial',
                 'number of modules axial', 'number of modules transaxial',
                 'number of submodules axial', 'number of submodules transaxial',
                 'number of crystals axial', 'number of crystals transaxial']
    res_list = search_txt_file(geom_file_path, tags_list, 6)
    res_list = np.array(res_list).astype(int)
    # print(res_list)

    layer_0 = res_list[:, 0]
    print(layer_0)

    number_of_crystals_per_ring = layer_0[0] * layer_0[3] * layer_0[5] * layer_0[7]
    print(number_of_crystals_per_ring)

    aa = determine_layer(np.array(1505))
    print(aa)
    # sys.exit()

    a = np.ravel_multi_index([2, 2, 3, 1505], [3, 24, 7, 16 * 110], order='C')
    print(a)


    # todo: Find out, which layer




    sys.exit()

    return 0


def search_txt_file(txt_file_path, tags_list, n_entries):
    txt_file = open(txt_file_path, 'r')
    out_list = []
    order_list = []
    for line in txt_file:
        for ii in range(len(tags_list)):
            if line.startswith(tags_list[ii] + ':'):
                without_whitespace = ''.join(line[len(tags_list[ii]) + 1:].split())
                separated = split(',|#', without_whitespace)
                # print(separated[:n_entries])
                out_list.append(separated[:n_entries])
                order_list.append(ii)

    out_list = [out_list[i] for i in np.argsort(np.array(order_list))]

    if len(out_list) != len(tags_list):
        print('Warning: Not all tags were found.')
    return out_list


def determine_layer(layer_id):
    y_rep = 16
    z_rep = 110

    layer_number = np.zeros(layer_id.shape, dtype=int)
    layer_number[layer_id < (y_rep * z_rep)] = 1
    layer_number[(layer_id >= (y_rep * z_rep)) & (layer_id < (2 * y_rep * z_rep))] = 2
    layer_number[layer_id >= (2 * y_rep * z_rep)] = 3

    return layer_number


if __name__ == "__main__":
    main()
