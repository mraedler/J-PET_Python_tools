"""
Analyze the processes

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.constants import electron_mass, speed_of_light, electron_volt, hbar, fine_structure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    npz_file_primaries = np.load(sys.path[0] + '/data/metadata_primaries.npz')
    # print(npz_file_primaries.files)
    npz_file_secondaries = np.load(sys.path[0] + '/data/metadata_secondaries.npz')
    # print(npz_file_secondaries.files)

    is_parent = npz_file_secondaries['is_parent']
    is_parent = np.ones(is_parent.shape, dtype=bool)
    # print(np.sum(is_parent))
    # print(npz_file_secondaries['is_parent'])
    # print(npz_file_primaries['step_count'].size)

    # Concatenate both
    step_count = np.concatenate((npz_file_primaries['step_count'], npz_file_secondaries['step_count'][~is_parent]))
    compton_count = np.concatenate((npz_file_primaries['compton_count'], npz_file_secondaries['compton_count'][~is_parent]))
    rayleigh_count = np.concatenate((npz_file_primaries['rayleigh_count'], npz_file_secondaries['rayleigh_count'][~is_parent]))
    photoelectric_count = np.concatenate((npz_file_primaries['photoelectric_count'], npz_file_secondaries['photoelectric_count'][~is_parent]))

    energies_initial_compton = np.concatenate((npz_file_primaries['energies_initial_compton'], npz_file_secondaries['energies_initial_compton']))
    energies_deposited_compton = np.concatenate((npz_file_primaries['energies_deposited_compton'], npz_file_secondaries['energies_deposited_compton']))
    energies_rayleigh = np.concatenate((npz_file_primaries['energies_rayleigh'], npz_file_secondaries['energies_rayleigh']))
    energies_photoelectric = np.concatenate((npz_file_primaries['energies_photoelectric'], npz_file_secondaries['energies_photoelectric']))

    # print(step_count.size)

    #
    # step_count = npz_file_primaries['step_count']
    # compton_count = npz_file_primaries['compton_count']
    # rayleigh_count = npz_file_primaries['rayleigh_count']
    # photoelectric_count = npz_file_primaries['photoelectric_count']

    # print(step_count.size)

    n_only_compton = np.sum((compton_count >= 0) & (rayleigh_count == 0) & (photoelectric_count == 0))
    n_compton_rayleigh = np.sum((compton_count >= 0) & (rayleigh_count > 0) & (photoelectric_count == 0))
    n_photoelectric = np.sum(photoelectric_count > 0)

    print(n_only_compton)
    print(n_compton_rayleigh)
    print(n_photoelectric)
    print(n_only_compton + n_compton_rayleigh + n_photoelectric)


    # Energy loss distribution
    # energy_loss_compton(energies_initial_compton, energies_deposited_compton)
    # energy_histogram(energies_rayleigh)
    energy_histogram(energies_photoelectric)

    # print(step_count)
    # print(rayleigh_count)

    # print(np.sum(step_count))
    # print(np.sum(compton_count))
    # print(np.sum(rayleigh_count))
    # print(np.sum(photoelectric_count))
    # print(step_count.size)

    n_events = 10004324
    n_gammas = 2 * n_events

    h = np.bincount(photoelectric_count)
    # h[0] = n_gammas - np.sum(h)
    # print(h)

    # print(step_count.size)

    # n_events = 10004324
    # n_gammas = 2 * n_events

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.bar(np.arange(h.size), h, width=0.8)
    # ax.bar(0, n_gammas - np.sum(h), width=0.8)
    ax0.set_xlabel('Number of interactions')
    ax1.bar(np.arange(h.size), h, width=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of interactions')
    plt.show()


    return 0


def energy_loss_compton(energies_initial, energies_deposited):
    # Energy binning
    energy_bin_edges = np.arange(512 + 1) / 1000  # [MeV]
    energy_bin_centers = (energy_bin_edges[1:] + energy_bin_edges[:-1]) / 2

    # 2D histogram
    h_2d, _, _ = np.histogram2d(energies_initial, energies_deposited, bins=energy_bin_edges, density=False)
    compton_edge = energy_bin_edges * (1 - 1 / (1 + 2 * energy_bin_edges / (electron_mass * speed_of_light ** 2 / electron_volt / 1e6)))

    # Figure
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(h_2d.T,
                   extent=[energy_bin_edges[0], energy_bin_edges[-1], energy_bin_edges[0], energy_bin_edges[-1]],
                   norm=LogNorm(vmin=1, vmax=4e4), origin='lower')
    fig.colorbar(im, cax=cax, orientation='vertical', label='Count')
    ax.plot(energy_bin_edges, compton_edge, color='r')
    ax.set_facecolor(plt.get_cmap('viridis')(0.))
    ax.set_xlabel('Initial energy [MeV]')
    ax.set_ylabel('Energy loss [MeV]')
    # ax.set_xlim(.482, .512)
    # ax.set_ylim(.33, .36)
    plt.show()

    # Profile along chose energy
    idx = -2
    energy_gamma = np.array([energy_bin_centers[idx]])  # MeV
    klein_nishina = klein_nishina_energy(energy_bin_centers, energy_gamma)
    profile = h_2d[idx, :]
    profile /= np.trapz(profile, x=energy_bin_centers)

    fig, ax = plt.subplots()
    ax.plot(energy_bin_centers, profile, label='GATE')
    ax.plot(energy_bin_centers, klein_nishina, label='Klein-Nishina')
    ax.set_xlim(energy_bin_edges[0], energy_bin_edges[-1])
    ax.set_xlabel('Energy loss [MeV]')
    ax.set_ylabel('Normalized distribution')
    ax.legend()
    plt.show()
    return 0


def klein_nishina_energy(delta, e_gamma):
    # Electron rest energy
    e_0 = electron_mass * speed_of_light ** 2 / (1e6 * electron_volt)  # [MeV]

    # Reduced compton wavelength
    lambda_bar = hbar / (electron_mass * speed_of_light)

    # Mesh-grids for the formula
    delta_mesh, e_gamma_mesh = np.meshgrid(delta, e_gamma, indexing='ij')
    epsilon_mesh = e_gamma_mesh / e_0

    # Compton edge
    compton_edge = e_gamma_mesh * (1 - 1 / (1 + 2 * epsilon_mesh))

    # Differential cross-section with respect to energy
    term_1 = 1 - delta_mesh / e_gamma_mesh
    term_2 = e_gamma_mesh / (e_gamma_mesh - delta_mesh)
    term_3 = (delta_mesh ** 2 / epsilon_mesh ** 2 - 2 * e_gamma_mesh * delta_mesh * epsilon_mesh + 2 * delta_mesh ** 2 / epsilon_mesh) / (e_gamma_mesh - delta_mesh) ** 2
    cross_section = np.pi * fine_structure ** 2 * lambda_bar ** 2 / (e_gamma_mesh * epsilon_mesh) * (term_1 + term_2 + term_3) * np.heaviside(compton_edge - delta_mesh, .5)

    # Normalization (integral area)
    n_1 = 2 * (1 + epsilon_mesh) / (1 + 2 * epsilon_mesh) ** 2
    n_2 = np.log(1 + 2 * epsilon_mesh) / epsilon_mesh
    n_3 = (4 * epsilon_mesh - 2 * (1 + epsilon_mesh) * np.log(1 + 2 * epsilon_mesh)) / epsilon_mesh ** 3
    sigma = np.pi * fine_structure ** 2 * lambda_bar ** 2 * (n_1 + n_2 + n_3)

    # Check normalization explicitly
    # print(np.trapz(cross_section, x=delta, axis=0))
    # print(sigma[0])

    return cross_section / sigma


def energy_histogram(data_list, label_list):
    # Energy binning
    energy_bin_edges = np.arange(512 + 1) / 1000  # [MeV]
    energy_bin_centers = (energy_bin_edges[1:] + energy_bin_edges[:-1]) / 2
    energy_bin_width = energy_bin_edges[1:] - energy_bin_edges[:-1]

    # Figure
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    for ii in range(len(data_list)):
        h, _ = np.histogram(data_list[ii], bins=energy_bin_edges)
        ax.bar(energy_bin_centers, h, width=energy_bin_width, alpha=0.75, label=label_list[ii])
    ax.set_xlim(energy_bin_edges[0], energy_bin_edges[-1])
    # ax.set_yscale('log')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Count')
    ax.legend(loc='upper left')
    plt.show()
    return 0


if __name__ == '__main__':
    main()
