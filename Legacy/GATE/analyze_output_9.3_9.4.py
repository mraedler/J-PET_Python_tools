"""
Analyze the difference of the Gate output between versions 9.3 and 9.4

Author: Martin Rädler
"""
# Python libraries
import sys
from uproot import open as open_root

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array


def main():

    directory = '/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/9.3_9.4_tests/'

    root_file = open_root(directory + '2025-01-29_14-49-37/results_1.root')
    coincidences_struct_9_3 = load_or_convert_to_structured_array(root_file['MergedCoincidences'])

    # root_file = open_root(directory + '2025-01-29_14-49-15/results_1.root')
    root_file = open_root(directory + '2025-01-31_09-51-49/results_1.root')
    coincidences_struct_9_4 = load_or_convert_to_structured_array(root_file['MergedCoincidences'])

    print(coincidences_struct_9_3.size)
    print(coincidences_struct_9_4.size)

    print(coincidences_struct_9_3['gantryID1'][:10])
    print(coincidences_struct_9_3['gantryID2'][:10])

    print(coincidences_struct_9_4['gantryID1'][:10])
    print(coincidences_struct_9_4['gantryID2'][:10])

    return 0


if __name__ == "__main__":
    main()
