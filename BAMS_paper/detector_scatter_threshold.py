"""
Edit the sensitivity map of a CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.constants import electron_mass, electron_volt, speed_of_light, fine_structure, hbar
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


def main():
    e_0 = electron_mass * speed_of_light ** 2 / electron_volt / 1e3  # keV
    e_th = (7 - np.sqrt(17)) / 8 * e_0  # keV

    delta_e_1 = np.linspace(0, 2/3 * e_0, 100)

    delta_e_2_max = 2 * (e_0 - delta_e_1) ** 2 / (2 * (e_0 - delta_e_1) + e_0)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(delta_e_1, delta_e_1, label=r'$\Delta E_1$')
    ax.plot(delta_e_1, delta_e_2_max, label=r'$\Delta E_{2,\mathrm{max}}$')
    # ax.plot(delta_e_1, delta_e_1 + delta_e_2_max)
    ax.plot([delta_e_1[0], delta_e_1[-1]], [e_th, e_th], color='k', linestyle=':')
    ax.plot([delta_e_1[0], delta_e_1[-1]], [100, 100], color='k', linestyle=':')
    ax.set_xlim(delta_e_1[0], delta_e_1[-1])
    ax.set_ylim(delta_e_1[0], delta_e_1[-1])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel(r'$\Delta E_1$ [keV]')
    ax.set_ylabel(r'$\Delta E$ [keV]')
    ax.legend(loc='upper center', frameon=False)
    plt.show()

    return 0


def klein_nishina_energy_loss():
    e_0 = electron_mass * speed_of_light ** 2 / electron_volt / 1e3  # keV
    e_gamma = 1 * e_0
    epsilon = e_gamma / e_0
    lambda_bar_c = hbar / (electron_mass * speed_of_light)
    compton_edge = e_gamma * (1 - 1 / (1 + 2 * epsilon))
    scaling_factor = np.pi * fine_structure ** 2 * lambda_bar_c ** 2

    delta = np.linspace(0, e_gamma, 1000)

    norm = (2 * (1 + epsilon) / (1 + 2 * epsilon) ** 2 +
            np.log(1 + 2 * epsilon) / epsilon +
            (4 * epsilon - 2 * (1 + epsilon) * np.log(1 + 2 * epsilon)) / epsilon ** 3)
    print(norm)

    d_sigma_d_delta = np.zeros(delta.shape)

    below_compton_edge = delta < compton_edge
    delta_sub = delta[below_compton_edge]

    d_sigma_d_delta[below_compton_edge] = 1 / (e_gamma * epsilon) * \
        ((e_gamma - delta_sub) / e_gamma +
         e_gamma / (e_gamma - delta_sub) +
         (delta_sub ** 2 / epsilon ** 2 - 2 * e_gamma * delta_sub / epsilon + 2 * delta_sub ** 2 / epsilon) / (e_gamma - delta_sub) ** 2)

    print(np.trapz(d_sigma_d_delta, x=delta))

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # print(fig.get_figwidth(), fig.get_figheight())
    ax.plot(delta, scaling_factor * d_sigma_d_delta / (1e-28 * 1e-3))
    ax.fill_between(delta[delta <= 200], scaling_factor * d_sigma_d_delta[delta <= 200] / (1e-28 * 1e-3), alpha=1/3)
    ax.set_xlim(delta[0], delta[-1])
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    y_lim = ax.get_ylim()
    ax.plot([200, 200], y_lim, color='k', linestyle='--')
    ax.set_ylim(y_lim)
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_xlabel(r'Energy deposition $\Delta$ [keV]')
    ax.set_ylabel(r'Cross section $\mathrm{d}\sigma/\mathrm{d}\Delta$ [mb/keV]')

    # ax_twin = ax.twinx()
    # ax_twin.plot(delta, cumtrapz(d_sigma_d_delta, x=delta, initial=0) / norm, color='tab:orange')

    plt.show()

    return 0


if __name__ == "__main__":
    # main()
    klein_nishina_energy_loss()
