"""
Extract details about the simulation based on the .mac file

@author: Martin Rädler
"""
# Python libraries
import sys


def summarize_simulation_specifications():

    mac_file = open(sys.argv[1], 'r')
    read_lines = mac_file.readlines()

    data_dict = {'Brain scanner geometry': {},
                 'Digitizer': {},
                 'Simulation': {}}

    gate_brain_commands = [['/gate/brainScanner/geometry/setHeight', 'Length'],
                           ['/gate/brainModule1/placement/setTranslation', 'Radius'],
                           ['/gate/brainScanner/placement/setTranslation', 'Z shift'],
                           ['/gate/brainLayer_1/geometry/setXLength', 'Layer 1 thickness'],
                           ['/gate/brainLayer_2/geometry/setXLength', 'Layer 2 thickness']]

    gate_digitizer_commands = [['/gate/digitizerMgr/layer_2/SinglesDigitizer/MergedSingles/energyFraming/setMin', 'Energy cut']]

    gate_simulation_commands = [['/gate/application/setTimeStart', 'Start'],
                                ['/gate/application/setTimeStop', 'Stop']]

    for line in read_lines:

        for ii in range(len(gate_brain_commands)):
            if line.startswith(gate_brain_commands[ii][0]):
                data_dict['Brain scanner geometry'].update({gate_brain_commands[ii][1]: line[len(gate_brain_commands[ii][0])+1:-1]})

        for jj in range(len(gate_digitizer_commands)):
            if line.startswith(gate_digitizer_commands[jj][0]):
                data_dict['Digitizer'].update({gate_digitizer_commands[jj][1]: line[len(gate_digitizer_commands[jj][0])+1:-1]})

        for jj in range(len(gate_simulation_commands)):
            if line.startswith(gate_simulation_commands[jj][0]):
                data_dict['Simulation'].update({gate_simulation_commands[jj][1]: line[len(gate_simulation_commands[jj][0])+1:-1]})

    #
    print_sub_dict_to_terminal(data_dict, 'Brain scanner geometry')
    print_sub_dict_to_terminal(data_dict, 'Digitizer')
    print_sub_dict_to_terminal(data_dict, 'Simulation')

    return 0


def print_sub_dict_to_terminal(nested_dict, tag):
    sub_dict = nested_dict[tag]

    max_len = 0
    for key in sub_dict:
        if len(key) > max_len:
            max_len = len(key)

    print_string = '%' + '%d' % max_len + 's: %s'

    print('\n' + tag + '\n' + '=' * len(tag))
    for key in sub_dict:
        print(print_string % (key, sub_dict[key]))

    return 0


if __name__ == "__main__":
    summarize_simulation_specifications()
