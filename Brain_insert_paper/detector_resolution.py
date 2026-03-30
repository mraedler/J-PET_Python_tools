"""
Data visualization

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    tau = 1
    ell = 10

    t = np.linspace(-2 * tau, 2 * tau, 101)
    l = np.linspace(-ell, ell, 201)

    dt, dl = t[1] - t[0], l[1] - l[0]

    # extent = (t[0] - dt / 2, t[-1] + dt / 2, l[0] - dl / 2, l[-1] + dl / 2)
    extent = (l[0] - dl / 2, l[-1] + dl / 2, t[0] - dt / 2, t[-1] + dt / 2)

    # t_mesh, l_mesh = np.meshgrid(t, l, indexing='ij')
    l_mesh, t_mesh = np.meshgrid(l, t, indexing='ij')

    f = resolution_model(t_mesh, tau, l_mesh, ell)

    # fig, ax = plt.subplots()
    # # ax.plot(np.trapezoid(f, x=t, axis=1))
    # ax.plot(np.trapezoid(f, x=l, axis=0))
    # plt.show()

    x_ticks = [-ell, -ell / 2, 0, ell / 2, ell]

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    ax1.imshow(f.T, origin='lower', cmap='gray_r', extent=extent, alpha=0.5)
    ax1.plot([-ell, ell], [0, 0], color='k')
    ax1.plot([0, 0], [-tau, tau], color='k')
    ax1.set_aspect(1)
    ax1.set_xlim()
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([r'$-\ell$', r'$-\ell/2$', r'$0$', r'$\ell/2$', r'$\ell$'])
    ax1.set_yticks([-tau, 0, tau])
    ax1.set_yticklabels([r'$-\tau$', r'$0$', r'$\tau$'])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xlabel(r'$l$, $t$, D1, D2')
    # ax1.set_ylabel(r'$t$')

    for ll in x_ticks:
        ax0.plot(np.linspace(-1, 1, 101) + ll, f[l == ll, :].squeeze(), color='k')
    ax0.set_ylim([0, 5])
    ax0.set_yticks([])
    ax0.set_xticks([])

    for spine in ax0.spines.values():
        spine.set_visible(False)

    ax1.set_xlim(ax0.get_xlim())


    plt.show()

    return 0


def resolution_model(t, tau, l, ell):
    #
    f = np.zeros(t.shape)

    # Normalized coordinates
    t_n = np.abs(t) / tau
    l_n = np.abs(l) / ell

    plateau = t_n < l_n

    drop = (l_n <= t_n) & (t_n < 1)

    f[plateau] = 1
    f[drop] = (t_n[drop] - 1) / (l_n[drop] - 1)

    f /= tau * (l_n + 1)

    return f


if __name__ == "__main__":
    main()
