"""

"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt


def main():
    #
    a = 1
    r = 1

    s = np.linspace(-2 * r, 2 * r, num=4001)
    f = a * (np.heaviside(s + r, 0.5) - np.heaviside(s - r, 0.5))

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 6))
    p, = ax.plot(s, f, linewidth=3, color='k')

    ff = np.zeros(s.shape)
    for nn in [0, 1, 3, 5, 7]:
        if nn == 0:
            f_nn = a / 2
        else:
            f_nn = a * 2 / (np.pi * nn) * np.sin(np.pi * nn / 2)
        ff += f_nn * np.cos(2 * np.pi * nn / (4 * r) * s)

        ax.plot(s, ff, label='n = %d' % nn)

    ax.set_xlabel(r'$s$ [mm]')

    ax.set_xticks([-2*r, -r, 0, r, 2*r])
    ax.set_xticklabels([r'$-2R_i$', r'$-R_i$', r'$0$', r'$R_i$', r'$2R_i$'])
    ax.set_yticks([0, a/2, a])
    ax.set_yticklabels([r'$0$', r'$A/2$', r'$A$'])
    legend = ax.legend(loc='upper right', frameon=False, title=r'$f(s)$ up to')

    ax.legend([p], [r'$f(s)$'], loc='upper left', frameon=False)

    ax.add_artist(legend)

    plt.show()

    return 0


if __name__ == "__main__":
    main()
