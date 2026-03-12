"""
Analyze the utilization of the layers in the PET layers

Author: Martin Rädler
"""
# Python libraries
import sys

import numpy as np
from uproot import open as open_root

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array
from crystal_id import determine_layer


def main():
    root_path = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-04-11_10-03-38/results.root'
    root_file = open_root(root_path)
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'])

    # Determine the true coincidences
    true_coincidences = ((coincidences_struct['eventID1'] == coincidences_struct['eventID2'])
                         & (coincidences_struct['comptonCrystal1'] == 1)
                         & (coincidences_struct['comptonCrystal2'] == 1))

    equate_layer_utilization(coincidences_struct)
    equate_layer_utilization(coincidences_struct[true_coincidences])
    sys.exit()

    # Determine in which layer each coincidence occurs
    # layer_number1 = determine_layer(coincidences_struct['layerID1'])
    # layer_number2 = determine_layer(coincidences_struct['layerID2'])
    layer_number1 = determine_layer(coincidences_struct['layerID1'])[true_coincidences]
    layer_number2 = determine_layer(coincidences_struct['layerID2'])[true_coincidences]

    n_11 = np.sum((layer_number1 == 1) & (layer_number2 == 1))
    n_12 = np.sum(layer_number1 != layer_number2)
    n_22 = np.sum((layer_number1 == 2) & (layer_number2 == 2))

    p_1 = (np.sum(layer_number1 == 1) + np.sum(layer_number2 == 1)) / (layer_number1.size + layer_number2.size)

    tot = n_11 + n_12 + n_22
    # todo: Try to introduce a shielding factor to explain the observation
    print('p_1 = %1.3f' % p_1)
    print('11: %1.3f | %1.3f' % (n_11 / tot, p_1 ** 2))
    print('12: %1.3f | %1.3f' % (n_12 / tot, 2 * p_1 * (1 - p_1)))
    print('22: %1.3f | %1.3f' % (n_22 / tot, (1 - p_1) ** 2))

    # print(p_1 ** 2)
    # print(2 * p_1 * (1 - p_1))
    # print((1 - p_1) ** 2)

    return 0


def equate_layer_utilization(coincidences_struct):
    layer_number1 = determine_layer(coincidences_struct['layerID1'])
    layer_number2 = determine_layer(coincidences_struct['layerID2'])

    idx = np.arange(coincidences_struct.size)
    b_11, b_12, b_22, n_11, n_12, n_22 = categorize_coincidences(layer_number1, layer_number2, show_statistics=True)

    #
    idx_11_remove = np.random.choice(idx[b_11], size=n_11 - n_22, replace=False)
    idx_12_remove = np.random.choice(idx[b_12], size=n_12 - 2 * n_22, replace=False)

    b_select = np.ones(coincidences_struct.size, dtype=bool)
    b_select[idx_11_remove] = False
    b_select[idx_12_remove] = False

    print()
    categorize_coincidences(layer_number1[b_select], layer_number2[b_select], show_statistics=True)

    coincidences_struct_new = coincidences_struct[b_select]

    print('%1.1f %% removed.' % (100 * (1 - coincidences_struct_new.size / coincidences_struct.size)))
    return coincidences_struct_new


def categorize_coincidences(layer_number1, layer_number2, show_statistics=False):
    b_11 = (layer_number1 == 1) & (layer_number2 == 1)
    b_12 = layer_number1 != layer_number2
    b_22 = (layer_number1 == 2) & (layer_number2 == 2)

    n_11 = np.sum(b_11)
    n_12 = np.sum(b_12)
    n_22 = np.sum(b_22)

    if show_statistics:
        tot = n_11 + n_12 + n_22
        print('11: %1.1f %%' % (100 * n_11 / tot))
        print('12: %1.1f %%' % (100 * n_12 / tot))
        print('22: %1.1f %%' % (100 * n_22 / tot))

    return b_11, b_12, b_22, n_11, n_12, n_22


if __name__ == "__main__":
    main()
