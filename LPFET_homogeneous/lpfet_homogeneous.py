import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# For plotting of the molecule (if you don't need this you can delete Molecule.plot_hubbard_molecule
# and this import statements
import sys

sys.path.extend(['/mnt/c/Users/tinc9/Documents/CNRS-offline/', '../'])
import quantnbody as qnb  # Folder qnb has to be in the sys.path or installed as package.
import quantnbody_class_new as class_qnb
import essentials

COMPENSATION_1_RATIO = 0.75  # for the Molecule.update_v_hxc
COMPENSATION_MAX_ITER_HISTORY = 4


def casci_dimer(gamma, h, u, v_imp, embedded_mol):
    # Density in range [0, 2]
    if np.sum(gamma) == 0:
        return 0, 0
    P, v = qnb.tools.householder_transformation(gamma)
    h_tilde = P @ h @ P
    h_tilde_dimer = h_tilde[:2, :2]
    u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
    u_0_dimer[0, 0, 0, 0] += u
    h_tilde_dimer[0, 0] += v_imp
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()
    density_dimer = qnb.tools.build_1rdm_alpha(embedded_mol.WFT_0, embedded_mol.a_dagger_a)
    density = density_dimer[0, 0] * 2
    # print("--------\ndensity")
    # single_shot.print_matrix(density_dimer)
    # print(embedded_mol.eig_values)
    energy = embedded_mol.eig_values[0]
    return density, energy


class Molecule:
    def __init__(self, site_number, u, t, mu_ext, n_e_start=-1, description=''):
        self.description = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}" + description
        # Basic data about system
        self.Ns = site_number
        if n_e_start == -1:
            self.Ne = self.Ns - 1e-5
        else:
            self.Ne = n_e_start
        self.Ne_floor = np.nan
        self.Ne_ceil = np.nan

        # Parameters for the system
        self.u = u  # for the start only 1D array and not 4d tensor
        self.t = t  # 2D one electron Hamiltonian with all terms for i != j
        self.mu_ext = mu_ext
        self.mu_ks_floor = np.inf
        self.mu_ks_ceil = np.inf
        self.v_hxc_floor = np.inf
        self.v_hxc_ceil = np.inf

        # density matrix
        self.y_floor = np.zeros((self.Ns, self.Ns), dtype=np.float64)
        self.y_ceil = np.zeros((self.Ns, self.Ns), dtype=np.float64)

        # KS
        self.h_ks = np.array((), dtype=np.float64)
        self.wf_ks = np.array((), dtype=np.float64)  # Kohn-Sham wavefunciton
        self.epsilon_s = np.array((), dtype=np.float64)  # Kohn-Sham energies
        self.n_ks = np.array((), dtype=np.float64)  # Densities of KS sysyem

        # qnb objects
        # self.whole_mol = class_qnb.HamiltonianV2(self.Ns, self.Ne)
        self.embedded_mol = class_qnb.HamiltonianV2(2, 2)
        self.embedded_mol.build_operator_a_dagger_a()

        self.alpha = np.inf
        self.precalculated_values = dict()

    def precalculate_rdms(self):
        self.h_ks = essentials.generate_huckel_hamiltonian(self.Ns, 2, self.t)
        self.epsilon_s, self.wf_ks = np.linalg.eigh(self.h_ks)
        ei_val_index = None
        y = np.zeros((self.Ns, self.Ns), float)
        for i in range(0, self.Ns * 2 + 1, 2):
            ret_dict = dict()
            print('Building hamiltonian for ', self.Ns, i)
            if i == 0:

                ei_val_index = 0
                ret_dict['mu_ks'] = self.epsilon_s[ei_val_index]
                # ret_dict['mu_ks'] = - 10
            else:
                ind_gamma = i // 2 - 1
                ei_val_index = (i - 1) // 2
                ret_dict['mu_ks'] = self.epsilon_s[ei_val_index]
                temp1 = self.wf_ks[:, ind_gamma][np.newaxis]
                y += temp1.T @ temp1

            ret_dict['Ne'] = i
            ret_dict['y'] = y.copy()

            self.precalculated_values[i] = ret_dict

    def calculate_ks_fci(self, every_x=2):
        if every_x == 1:
            i_floor = int(np.floor(self.Ne))
            i_ceil = i_floor + every_x
        elif every_x == 2:
            i_floor = int((self.Ne - 1e-10) // 2 * 2)
            i_ceil = i_floor + every_x
            if i_floor == -2:
                i_floor = 0
        elif every_x == 4:
            ne = self.Ne
            if ne < 2:
                i_floor = 0
                i_ceil = 2
            elif ne > self.Ns * 2 - 2:
                i_ceil = self.Ns * 2
                i_floor = i_ceil - 2
            else:
                i_floor = int((self.Ne - 2 - 1e-10) // 4 * 4 + 2)
                i_ceil = i_floor + 4
        self.y_floor = self.precalculated_values[i_floor]['y']
        self.Ne_floor = self.precalculated_values[i_floor]['Ne']
        self.mu_ks_floor = self.precalculated_values[i_floor]['mu_ks']
        self.v_hxc_floor = -self.mu_ks_floor + self.mu_ext

        self.y_ceil = self.precalculated_values[i_ceil]['y']
        self.Ne_ceil = self.precalculated_values[i_ceil]['Ne']
        self.mu_ks_ceil = self.precalculated_values[i_ceil]['mu_ks']
        self.v_hxc_ceil = -self.mu_ks_ceil + self.mu_ext

        if (self.Ne_ceil - self.Ne_floor) != 0:
            self.alpha = (self.Ne - self.Ne_floor) / (self.Ne_ceil - self.Ne_floor)
        # else:
        #     self.alpha = 0.5
        print(f'\t{self.Ne:6.2f}{self.Ne_ceil:5d}, {self.Ne_floor:5d}{self.v_hxc_ceil:6.2f}{self.v_hxc_floor:6.2f}',
              end='')

    def pre_casci(self):
        density_floor, energy_floor = casci_dimer(self.y_floor, self.h_ks, self.u, -self.v_hxc_floor,
                                                  self.embedded_mol)
        density_ceil, energy_ceil = casci_dimer(self.y_ceil, self.h_ks, self.u, -self.v_hxc_ceil, self.embedded_mol)
        density_ceil *= self.Ns
        density_floor *= self.Ns
        alpha = -1.0
        if (self.Ne_ceil - self.Ne_floor - density_ceil + density_floor) != 0:
            alpha = (density_floor - self.Ne_floor) / (self.Ne_ceil - self.Ne_floor - density_ceil + density_floor)

            if 0 < alpha < 1:
                self.alpha = alpha
        print(f'{alpha:6.2f}{self.alpha:6.2f}', end='')

    def CASCI(self):

        density_floor, energy_floor = casci_dimer(self.y_floor, self.h_ks, self.u, -self.v_hxc_floor,
                                                  self.embedded_mol)
        density_ceil, energy_ceil = casci_dimer(self.y_ceil, self.h_ks, self.u, -self.v_hxc_ceil, self.embedded_mol)

        n_e_new = self.Ns * (self.alpha * density_ceil + (1 - self.alpha) * density_floor)
        self.Ne = n_e_new
        print(f"{density_floor:6.2f}{density_ceil:6.2f} -> {n_e_new:6.2f}")
        # alpha_new = (n_e_new - self.Ne_floor) / (self.Ne_ceil - self.Ne_floor)
        # print("new alpha is: ", alpha_new)
        # self.alpha = alpha_new


def calculate_from_v_ext(regime=4):
    regime = 4
    u_param = 5
    result_list = []
    num_points = 120
    mol1 = Molecule(10, u_param, 1, 0, -1)
    mol1.precalculate_rdms()
    missing_list = []
    xrange = np.linspace(-5, 7.5, num_points)
    for ind1, ext_pot in enumerate(xrange):
        print(f'External potential: {ext_pot} ({ind1 / num_points * 100:.2f}%)')
        mol1.mu_ext = ext_pot
        old_density = np.nan
        ii = 0
        n_e_list = []
        mol1.Ne = mol1.Ns - 1e-5
        for ii in range(60):
            mol1.calculate_ks_fci(regime)
            mol1.pre_casci()
            mol1.CASCI()
            if (len(n_e_list) // 10 == 0) and len(n_e_list) > 11 and (abs(n_e_list[-1] - n_e_list[-2]) > 0.02):
                print(ii, np.average(n_e_list[-8:]), n_e_list[-8:], mol1.Ne)
                mol1.Ne = np.average(n_e_list[-8:])
            n_e_list.append(mol1.Ne)
            new_density = mol1.Ne
            if abs(new_density - old_density) < 1e-5:
                print('\t converged:', new_density, old_density)
                result_list.append([ext_pot, mol1.Ne, mol1.Ne_floor, mol1.Ne_ceil, mol1.mu_ks_floor, mol1.mu_ks_ceil,
                                    mol1.v_hxc_floor, mol1.v_hxc_ceil, mol1.alpha, ii])
                break
            old_density = new_density
        else:
            missing_list.append([ind1, ext_pot])

    result_list = np.array(result_list)
    rl = np.array([tuple(j) for j in result_list],
                  dtype=[('mu_ext', float), ('Ne', float), ('Ne_floor', float), ('Ne_ceil', float),
                         ('mu_ks_floor', float), ('mu_ks_ceil', float), ('v_hxc_floor', float),
                         ('v_hxc_ceil', float), ('alpha', float), ('iter_num', int)])

    previous_ind = -1
    missing_list_v2 = []
    delta = (xrange[1] - xrange[0]) / 2
    for i in range(len(missing_list)):
        if missing_list[i][0] == previous_ind + 1:
            missing_list_v2[-1][1] = missing_list[i][1] + delta
        else:
            missing_list_v2.append([missing_list[i][1] - delta, missing_list[i][1] + delta])
        previous_ind = missing_list[i][0]
    comment = f"_u-{u_param}"
    fig, ax = plt.subplots(4, 1, sharex='col', figsize=(10, 13))
    ax[0].title.set_text('Number of electrons')
    ax[0].scatter(rl['mu_ext'], rl['Ne'], c='k', label='Ne', s=3, marker='x')
    ax[0].scatter(rl['mu_ext'], rl['Ne_floor'], marker='x', s=3, label='Ne of floor determinant')
    ax[0].scatter(rl['mu_ext'], rl['Ne_ceil'], marker='x', s=3, label='Ne of ceil determinant')
    ax[0].scatter(rl['mu_ext'], rl['Ne'], marker='x', s=3, c='k')
    ax[0].legend()

    ax[1].title.set_text('Alpha')
    ax[1].scatter(rl['mu_ext'], rl['alpha'], marker='x', s=3, c='k')

    ax[2].title.set_text('Potentials')
    ax[2].scatter(rl['mu_ext'], rl['mu_ks_floor'], marker='x', s=3, label='mu_ks of floor determinant')
    ax[2].scatter(rl['mu_ext'], rl['mu_ks_ceil'], marker='x', s=3, label='mu_ks of ceil determinant')
    ax[2].scatter(rl['mu_ext'], rl['v_hxc_floor'], marker='x', s=3, label='v_hxc of floor determinant')
    ax[2].scatter(rl['mu_ext'], rl['v_hxc_ceil'], marker='x', s=3, label='v_hxc of ceil determinant')

    ax[2].legend()

    ax[3].title.set_text("# iterations for convergence")
    ax[3].scatter(rl['mu_ext'], rl['iter_num'], marker='x', s=3)

    ax[3].set_xlabel('External potential mu_ext')
    ax[0].set_ylabel('Ne')
    ax[1].set_ylabel('Alpha')
    ax[2].set_ylabel('mu or v')
    ax[3].set_ylabel('#')

    for i in range(4):
        box = ax[i].get_position()
        ax[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        if i in [0, 2]:
            ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')

        for line in missing_list_v2:
            ax[i].axvspan(line[0], line[1], alpha=0.15, color='red')

        ax[i].grid(color='#dedede', linestyle='-', linewidth=0.5)
    plt.savefig(f'results/delta-{regime}_combined{comment}.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.title.set_text('Number of electrons')
    ax.scatter(rl['mu_ext'], rl['Ne'], c='k', label='Ne', s=3, marker='x')
    ax.scatter(rl['mu_ext'], rl['Ne_floor'], marker='x', s=3, label='Ne of floor determinant')
    ax.scatter(rl['mu_ext'], rl['Ne_ceil'], marker='x', s=3, label='Ne of ceil determinant')
    ax.scatter(rl['mu_ext'], rl['Ne'], marker='x', s=3, c='k')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    for line in missing_list_v2:
        ax.axvspan(line[0], line[1], alpha=0.15, color='red')
    ax.grid(color='#dedede', linestyle='-', linewidth=0.5)
    ax.set_xlabel('External potential mu_ext')
    ax.set_ylabel('Ne')
    plt.savefig(f'results/delta-{regime}_Ne{comment}.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.title.set_text('Alpha')
    ax.scatter(rl['mu_ext'], rl['alpha'], marker='x', s=10, c='k')
    for line in missing_list_v2:
        ax.axvspan(line[0], line[1], alpha=0.15, color='red')
    ax.grid(color='#dedede', linestyle='-', linewidth=0.5)
    ax.set_xlabel('External potential mu_ext')
    ax.set_ylabel('Alpha')
    plt.savefig(f'results/delta-{regime}_alpha{comment}.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    ax.title.set_text('Potentials')
    ax.scatter(rl['mu_ext'], rl['mu_ks_floor'], marker='o', s=15, label='mu_ks of floor determinant', alpha=0.7,
               linewidth=1)
    ax.scatter(rl['mu_ext'], rl['mu_ks_ceil'], marker='x', s=15, label='mu_ks of ceil determinant', alpha=0.7,
               linewidth=1)
    ax.scatter(rl['mu_ext'], rl['v_hxc_floor'], marker='o', s=15, label='v_hxc of floor determinant', alpha=0.7,
               linewidth=1)
    ax.scatter(rl['mu_ext'], rl['v_hxc_ceil'], marker='x', s=15, label='v_hxc of ceil determinant', alpha=0.7,
               linewidth=1)
    # ax.scatter(rl['mu_ext'], rl['v_hxc_ceil'] - rl['v_hxc_floor'], marker='x', s=3, label=' delta v_hxc ')
    # ax.scatter(rl['mu_ext'], rl['mu_ks_ceil'] - rl['mu_ks_floor'], marker='x', s=3, label=' delta mu_ks ')
    ax.set_xlabel('External potential mu_ext')
    ax.set_ylabel('mu or v')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    for line in missing_list_v2:
        ax.axvspan(line[0], line[1], alpha=0.15, color='red')
    ax.grid(color='#dedede', linestyle='-', linewidth=0.5)
    plt.savefig(f'results/delta-{regime}_potentials{comment}.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    ax.title.set_text("# iterations for convergence")
    ax.scatter(rl['mu_ext'], rl['iter_num'], marker='x', s=10)
    ax.set_ylabel('#')
    ax.set_xlabel('External potential mu_ext')
    for line in missing_list_v2:
        ax.axvspan(line[0], line[1], alpha=0.15, color='red')
    ax.grid(color='#dedede', linestyle='-', linewidth=0.5)
    plt.savefig(f'results/delta-{regime}_iter-num{comment}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    calculate_from_v_ext(regime=4)
