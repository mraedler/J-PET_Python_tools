"""
Various utility functions

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from decimal import Decimal


"""
Print fit parameters with their errors
"""


def print_parameters(parameters, errors, names):
    for ii in range(len(parameters)):
        # Get exponent and mantissa of the parameter
        par_exp = get_float_exponent(parameters[ii])
        par_man = get_float_mantissa(parameters[ii])

        # Get exponent and mantissa of the error
        err_exp = get_float_exponent(errors[ii])
        err_man = get_float_mantissa(errors[ii])

        specifier = names[ii] + ' = %1.' + ('%d' % np.abs(err_exp - par_exp)) + 'f(%d)x10^%d'
        print(specifier % (par_man, round(float(err_man)), par_exp))
        # print('%1.3e' % parameters[ii])
        # print('%1.3e' % errors[ii])

    return 0


def get_float_exponent(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def get_float_mantissa(number):
    return Decimal(number).scaleb(-get_float_exponent(number)).normalize()


"""
Extent for plt.imshow(...)
"""


def get_extent(x, y):
    return (3 * x[0] - x[1]) / 2, (3 * x[-1] - x[-2]) / 2, (3 * y[0] - y[1]) / 2, (3 * y[-1] - y[-2]) / 2


"""
Data filters
"""


def get_gantry_filter(coincidences_struct, tag):
    if tag == 'TOT':
        gantry_filter = np.ones(coincidences_struct.size, dtype=bool)
    elif tag == 'TBTB':
        gantry_filter = (coincidences_struct['gantryID1'] == 0) & (coincidences_struct['gantryID2'] == 0)
    elif tag == 'BB':
        gantry_filter = (coincidences_struct['gantryID1'] == 1) & (coincidences_struct['gantryID2'] == 1)
    elif tag == 'TBB':
        gantry_filter = coincidences_struct['gantryID1'] != coincidences_struct['gantryID2']
    else:
        sys.exit('Error: unknown gantry filter tag.')
    return gantry_filter, tag


def get_true_filter(coincidences_struct, filter_true=True):
    if filter_true:
        # true_filter = ((coincidences_struct['eventID1'] == coincidences_struct['eventID2'])
        #                & (coincidences_struct['comptonCrystal1'] == 1)
        #                & (coincidences_struct['comptonCrystal2'] == 1))
        true_filter = ((coincidences_struct['eventID1'] == coincidences_struct['eventID2'])
                       & (coincidences_struct['comptonCrystal1'] == 1)
                       & (coincidences_struct['comptonCrystal2'] == 1)
                       & (coincidences_struct['RayleighCrystal1'] == 0)
                       & (coincidences_struct['RayleighCrystal2'] == 0))
        tag = '_true'
    else:
        true_filter = np.ones(coincidences_struct.size, dtype=bool)
        tag = ''

    return true_filter, tag
