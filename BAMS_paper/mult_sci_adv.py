"""
Analyze the multiplicity from the Science Advances article
Moskal et al., Sci. Adv. 10, eadp2840 (2024)

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.loadtxt(sys.path[0] + '/mult_sci_adv.csv', delimiter=',', skiprows=1)
    multiplicity = data[:, 0]
    counts = data[:, 1]

    ratio = counts[:-1] / counts[1:]
    multiplicity_mid = (multiplicity[:-1] + multiplicity[1:]) / 2

    fig, ax = plt.subplots()
    # ax.bar(multiplicity, counts)
    ax.stairs(counts, edges=np.arange(data.shape[0] + 1) + 1 / 2)
    ax.set_yscale('log')
    ax_twin = ax.twinx()
    ax_twin.plot(multiplicity_mid, ratio)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
