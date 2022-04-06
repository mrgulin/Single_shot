import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.optimize as sc_opt
from datetime import datetime
# For plotting of the molecule (if you don't need this you can delete Molecule.plot_hubbard_molecule
# and this import statements
import pandas as pd
import networkx as nx
import sys
from sklearn.linear_model import LinearRegression
from typing import Union
import typing

sys.path.extend(['/mnt/c/Users/tinc9/Documents/CNRS-offline/', '../'])
import essentials
import Quant_NBody  # Folder Quant_NBody has to be in the sys.path or installed as package.
import Quant_NBody.class_Quant_NBody as class_Quant_NBody
from essentials import OUTPUT_SEPARATOR, OUTPUT_FORMATTING_NUMBER, print_matrix, generate_1rdm

COMPENSATION_1_RATIO = 0.5  # for the Molecule.update_v_hxc
COMPENSATION_MAX_ITER_HISTORY = 4
COMPENSATION_5_FACTOR = 1
COMPENSATION_5_FACTOR2 = 0.5
ITERATION_NUM = 0


def change_indices(array_inp: np.array, site_id: int):
    array = np.copy(array_inp)
    if site_id != 0:
        # We have to move impurity on the index 0
        if array_inp.ndim == 2:
            array[:, [0, site_id]] = array[:, [site_id, 0]]
            array[[0, site_id], :] = array[[site_id, 0], :]
        elif array_inp.ndim == 1:
            array[[0, site_id]] = array[[site_id, 0]]
    return array


def log_calculate_ks_decorator(func):
    """
    Wrapper that adds data to Molecule.report_string for method calculate_ks. Doesn't affect behaviour of the program!!
    :param func:  calculate_ks function
    :return: nested function
    """

    def wrapper_func(self):
        ret_val = func(self)

        self.report_string += f"\tEntered calculate_ks\n"

        self.report_string += f"\t\tHartree exchange correlation potential\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.v_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + '\n'

        self.report_string += "\t\tv_s\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.v_s]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1)

        temp1 = print_matrix(self.h_ks, ret=True).replace('\n', '\n\t\t')[:-2]
        self.report_string += f'\n\t\th_ks\n\t\t' + temp1 + '\t\t----Diagonalization---'

        temp1 = print_matrix(self.wf_ks, ret=True).replace('\n', '\n\t\t')[:-2]
        self.report_string += f'\n\t\tKS wave function\n\t\t' + temp1

        self.report_string += "\t\tKohn Sham energy\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.epsilon_s]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1)

        temp1 = print_matrix(self.y_a, ret=True).replace('\n', '\n\t\t')[:-2]
        self.report_string += f'\n\t\t1RDM per spin\n\t\t' + temp1

        self.report_string += "\t\tKS density\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.n_ks]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + '\n'
        return ret_val

    return wrapper_func


def log_add_parameters_decorator(func):
    def wrapper_func(self, u, t, v_ext, equiv_atom_group_list, v_term_repulsion_ratio):
        ret_val = func(self, u, t, v_ext, equiv_atom_group_list, v_term_repulsion_ratio)
        self.report_string += f"add_parameters:\n\tU_iiii\n\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.u]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1)
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.v_ext]
        self.report_string += f'\n\tv_ext\n\t' + f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"
        temp1 = print_matrix(self.t, ret=True).replace('\n', '\n\t')[:-1]
        self.report_string += f'\n\tt\n\t' + temp1
        return ret_val

    return wrapper_func


class Molecule:
    def __init__(self, site_number, electron_number, description=''):
        self.description = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}" + description
        # Basic data about system
        self.Ns = site_number
        self.Ne = electron_number

        # Parameters for the system
        self.u = np.array((), dtype=np.float64)  # for the start only 1D array and not 4d tensor
        self.u4d = np.array((), dtype=np.float64)
        self.t = np.array((), dtype=np.float64)  # 2D one electron Hamiltonian with all terms for i != j
        self.v_ext = np.array((), dtype=np.float64)  # 1D one electron Hamiltonian with all terms for i == j
        self.equiv_atom_groups = dict()
        self.v_term = False

        # density matrix
        self.y_a = np.array((), dtype=np.float64)  # y --> gamma, a --> alpha so this indicates it is only per one spin

        # KS
        self.v_s = np.zeros(self.Ns, dtype=np.float64)  # Kohn-Sham potential
        self.h_ks = np.array((), dtype=np.float64)
        self.wf_ks = np.array((), dtype=np.float64)  # Kohn-Sham wavefunciton
        self.epsilon_s = np.array((), dtype=np.float64)  # Kohn-Sham energies
        self.n_ks = np.array((), dtype=np.float64)  # Densities of KS sysyem
        self.v_hxc = np.zeros(self.Ns, dtype=np.float64)  # Hartree exchange correlation potential
        self.imp_potential = np.zeros(self.Ns, dtype=np.float64)

        # Energy
        self.kinetic_contributions = np.zeros(self.Ns, dtype=np.float64)
        # This is where \hat t^{(i)} are going to be written
        self.onsite_repulsion = np.zeros(self.Ns, dtype=np.float64)
        self.v_term_repulsion = np.zeros(self.Ns, dtype=np.float64)
        self.energy_contributions = tuple()
        self.per_site_energy = np.array(())

        # Quant_NBody objects
        # self.whole_mol = class_Quant_NBody.QuantNBody(self.Ns, self.Ne)
        self.embedded_mol = class_Quant_NBody.QuantNBody(2, 2)
        self.embedded_mol.build_operator_a_dagger_a()

        self.report_string = f'Object with {self.Ns} sites and {self.Ne} electrons\n'

        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.v_hxc_progress = []

        self.iteration_i = 0
        self.oscillation_correction_dict = dict()

        self.h_tilde_dimer = dict()

        self.compensation_ratio_dict = dict()

    @log_add_parameters_decorator
    def add_parameters(self, u, t, v_ext, equiv_atom_group_list,
                       v_term_repulsion_ratio: typing.Union[bool, float] = False):
        if len(u) != self.Ns or len(t) != self.Ns or len(v_ext) != self.Ns:
            raise f"Problem with size of matrices: U={len(u)}, t={len(t)}, v_ext={len(v_ext)}"
        self.u = u
        self.t = t
        self.v_ext = v_ext
        self.u4d = np.zeros((self.Ns, self.Ns, self.Ns, self.Ns))
        for i in range(self.Ns):
            self.u4d[i, i, i, i] = self.u[i]
            if v_term_repulsion_ratio:
                self.v_term = True
                for a in range(len(self.t)):
                    for b in range(len(self.t)):
                        if self.t[a, b] != 0:
                            self.u4d[a, a, b, b] = v_term_repulsion_ratio * 0.5 * (u[a] + u[b])
                            self.u4d[b, b, a, a] = v_term_repulsion_ratio * 0.5 * (u[a] + u[b])
        if 0 not in equiv_atom_group_list[0]:
            for i in range(len(equiv_atom_group_list)):
                if 0 in equiv_atom_group_list[i]:
                    temp_var = equiv_atom_group_list[0].copy()
                    equiv_atom_group_list[0] = equiv_atom_group_list[i].copy()
                    equiv_atom_group_list[i] = temp_var
                    break
        for index, item in enumerate(equiv_atom_group_list):
            self.equiv_atom_groups[index] = tuple(item)
            self.compensation_ratio_dict[index] = COMPENSATION_5_FACTOR

    def self_consistent_loop(self, num_iter=10, tolerance=0.0001, overwrite_output="",
                             oscillation_compensation: typing.Union[int, typing.List[int]] = 0, v_hxc_0=None):
        self.report_string += "self_consistent_loop:\n"
        old_density = np.inf
        old_v_hxc = np.inf
        i = 0
        for i in range(num_iter):
            self.iteration_i = i
            self.report_string += f"Iteration # = {i}\n"
            self.calculate_ks()
            self.density_progress.append(self.n_ks.copy())
            self.casci(oscillation_compensation, v_hxc_0)
            self.v_hxc_progress.append(self.v_hxc.copy())
            print(f"\nLoop {i}", end=', ')
            mean_square_difference_density = np.average(np.square(self.n_ks - old_density))
            max_difference_v_hxc = np.max(np.abs(self.v_hxc - old_v_hxc))

            self.log_scl(old_density, mean_square_difference_density, i, tolerance, num_iter)

            if mean_square_difference_density < tolerance and max_difference_v_hxc < 0.01:
                break
            old_density = self.n_ks
            old_v_hxc = self.v_hxc.copy()
        self.report_string += f'Final Hxc chemical potential:\n'
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.v_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"
        if overwrite_output:
            conn = open(overwrite_output, "w", encoding="UTF-8")
        else:
            conn = open(f"results/{self.description}_log.dat", "w", encoding="UTF-8")
        conn.write(self.report_string)
        conn.close()
        return i

    def find_solution_as_root(self, starting_approximation=None):
        if starting_approximation is None:
            starting_approximation = np.zeros(len(self.equiv_atom_groups) - 1, float)
            for group_key, group_tuple in self.equiv_atom_groups.items():
                if group_key != 0:
                    group_element_site = group_tuple[0]
                    starting_approximation[group_key - 1] = (self.v_ext[0] - self.v_ext[group_element_site]) * 0.5
        elif len(starting_approximation) != len(self.equiv_atom_groups) - 1:
            raise Exception('Wrong length of the starting approximation')
        model = scipy.optimize.root(cost_function_whole, starting_approximation,
                                    args=(self,), options={'fatol': 2e-3}, method='df-sane')
        if not model.success or np.sum(np.square(model.fun)) > 0.01:
            print("Didn't converge :c ")
            return False
        v_hxc = model.x
        print(model.fun)
        print(model)
        return v_hxc

    @log_calculate_ks_decorator
    def calculate_ks(self):
        self.v_s = self.v_hxc + self.v_ext
        self.h_ks = self.t + np.diag(self.v_s)
        self.epsilon_s, self.wf_ks = np.linalg.eigh(self.h_ks, 'U')
        self.y_a = generate_1rdm(self.Ns, self.Ne, self.wf_ks)
        self.n_ks = np.copy(self.y_a.diagonal())

    def casci(self, oscillation_compensation=0, v_hxc_0=None):
        self.report_string += "\tEntered CASCI\n"
        mu_imp_first = np.nan
        first_iteration = True
        for site_group in self.equiv_atom_groups.keys():
            self.report_string += f"\t\tGroup {site_group} with sites {self.equiv_atom_groups[site_group]}\n"
            site_id = self.equiv_atom_groups[site_group][0]

            # Householder transforms impurity on index 0 so we have to make sure that impurity is on index 0:
            y_a_correct_imp = change_indices(self.y_a, site_id)
            t_correct_imp = change_indices(self.t, site_id)
            v_s_correct_imp = change_indices(self.v_s, site_id)

            P, v = Quant_NBody.householder_transformation(y_a_correct_imp)
            h_tilde = P @ (t_correct_imp + np.diag(v_s_correct_imp)) @ P

            h_tilde_dimer = h_tilde[:2, :2]
            u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
            u_0_dimer[0, 0, 0, 0] += self.u[site_id]
            mu_imp = self.v_hxc[[site_id]]  # Double parenthesis so I keep array, in future this will be list of
            # indices for block householder

            self.log_casci(site_id, y_a_correct_imp, P, v, h_tilde, h_tilde_dimer)
            self.h_tilde_dimer[site_group] = h_tilde_dimer

            sol = sc_opt.root_scalar(cost_function_casci_root,
                                     args=(self.embedded_mol, h_tilde_dimer, u_0_dimer, self.n_ks[site_id]),
                                     bracket=[-0.1, 15], method='brentq', options={'xtol': 1e-6})
            mu_imp, function_calls = sol.root, sol.function_calls
            if first_iteration:
                if 0 not in self.equiv_atom_groups[site_group]:
                    raise Exception("Unexpected behaviour: First impurity site should have been the 0th site")
                mu_imp_first = mu_imp

            delta_mu_imp = mu_imp - mu_imp_first
            self.report_string += f'\t\t\tOptimized chemical potential mu_imp: {mu_imp}, done in {function_calls}' \
                                  f' iterations\n\t\t\tRelative change from impurity 0: {delta_mu_imp}'
            if v_hxc_0 is None:
                # This means that if we keep v_hxc_0 equal to None,
                # the algorithm does mu_imp -> v_hxc (mu_imp - mu_imp0 + mu_imp0)
                v_hxc_0 = mu_imp_first
            self.update_v_hxc(site_group, v_hxc_0 + delta_mu_imp, oscillation_compensation)

            two_rdm = self.embedded_mol.calculate_2rdm_fh(index=0)
            on_site_repulsion_i = two_rdm[0, 0, 0, 0] * u_0_dimer[0, 0, 0, 0]
            v_term_repulsion_i = np.sum(two_rdm * u_0_dimer) - on_site_repulsion_i
            for every_site_id in self.equiv_atom_groups[site_group]:
                self.kinetic_contributions[every_site_id] = 2 * h_tilde[1, 0] * self.embedded_mol.one_rdm[1, 0]
                self.onsite_repulsion[every_site_id] = on_site_repulsion_i
                self.imp_potential[every_site_id] = mu_imp
                self.v_term_repulsion[every_site_id] = v_term_repulsion_i

            first_iteration = False

        self.report_string += f'\t\tHxc chemical potential in the end of a cycle:\n\t\t'
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.v_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"

    def update_v_hxc(self, site_group, mu_imp, oscillation_compensation):
        global COMPENSATION_5_FACTOR
        if type(oscillation_compensation) == int:
            oscillation_compensation = [oscillation_compensation]
        elif len(self.v_hxc_progress) < 2:
            index = self.equiv_atom_groups[site_group][0]
            if len(self.v_hxc_progress) == 0:
                mu_minus_1 = self.v_hxc[index]
            else:
                mu_minus_1 = self.v_hxc_progress[-1][index]
            if 5 in oscillation_compensation:
                new_mu_imp = mu_minus_1
                new_mu_imp += np.tanh((mu_imp - mu_minus_1)) / COMPENSATION_5_FACTOR2
                print(f'(({site_group}): {mu_minus_1:.2f} {new_mu_imp - mu_minus_1:.2f} {mu_imp - mu_minus_1:.2f})',
                      end=', ')
                mu_imp = new_mu_imp
        else:
            index = self.equiv_atom_groups[site_group][0]
            mu_minus_2 = self.v_hxc_progress[-2][index]
            mu_minus_1 = self.v_hxc_progress[-1][index]

            if mu_imp - mu_minus_1 > 0:
                f_counter = np.argmin
                f_same = np.argmax
            else:
                f_counter = np.argmax
                f_same = np.argmin
            cur_iter_num = len(self.v_hxc_progress)
            pml = [0]
            for ind1 in range(max(cur_iter_num - COMPENSATION_MAX_ITER_HISTORY, 0), cur_iter_num):
                pml.append(self.v_hxc_progress[ind1][index])
            mu_counter = f_counter(pml)
            mu_same = f_same(pml)
            if 5 in oscillation_compensation:
                new_mu_imp = mu_minus_1
                if (mu_minus_2 - mu_minus_1) * (mu_minus_1 - mu_imp) < 0:
                    self.compensation_ratio_dict[index] += 0.5
                else:
                    self.compensation_ratio_dict[index] = max(self.compensation_ratio_dict[index] - 0.1, 1)
                new_mu_imp += np.tanh((mu_imp - mu_minus_1)) / self.compensation_ratio_dict[index]
                print(f'(({site_group}): {mu_minus_1:.2f} {new_mu_imp - mu_minus_1:.2f} {mu_imp - mu_minus_1:.2f},'
                      f' {self.compensation_ratio_dict[index]:.1f})',
                      end=', ')
                mu_imp = new_mu_imp
            if 1 in oscillation_compensation:
                if (mu_minus_2 - mu_minus_1) * (mu_minus_1 - mu_imp) < 0 and \
                        abs(mu_minus_2 - mu_minus_1) * 0.75 < abs(mu_minus_1 - mu_imp):
                    # First statement means that potential correction turned direction and second means that it is large
                    new_mu_imp = mu_minus_1 + (mu_imp - mu_minus_1) * COMPENSATION_1_RATIO
                    print(f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f}', end=', ')
                    mu_imp = new_mu_imp
                    self.oscillation_correction_dict[(self.iteration_i, index)] = (
                        mu_minus_2, mu_minus_1, mu_imp, new_mu_imp)
            if 2 in oscillation_compensation:

                if (abs(pml[mu_counter] - pml[mu_same]) * 0.5 < abs(mu_minus_1 - mu_imp)) and \
                        ((mu_counter - mu_same) > 0):
                    # First statement means that potential correction turned direction and second means that it is large
                    new_mu_imp = mu_minus_1 + (mu_imp - mu_minus_1) * COMPENSATION_1_RATIO
                    print(f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f}', end=', ')
                    mu_imp = new_mu_imp
                    self.oscillation_correction_dict[(self.iteration_i, index)] = (
                        mu_minus_2, mu_minus_1, mu_imp, new_mu_imp)
            if 3 in oscillation_compensation:
                pml2 = np.array(pml)
                x_data = np.arange(len(pml2)).reshape(-1, 1)
                reg = LinearRegression().fit(x_data, pml2)
                r2 = reg.score(x_data, pml2)
                factor1 = r2  # np.exp((r2 - 1) * 3)
                predicted = reg.predict([x_data[-1]])[0]
                new_mu_imp = factor1 * mu_imp + (1 - factor1) * predicted
                print(f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f} ({r2}, {factor1})', end=', ')
                mu_imp = new_mu_imp

        for every_site_id in self.equiv_atom_groups[site_group]:
            self.v_hxc[every_site_id] = mu_imp

    def calculate_energy(self, silent=False):
        per_site_array = np.zeros(self.Ns, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),
                                                  ('v_term', float)])
        per_site_array['kin'] = self.kinetic_contributions
        per_site_array['v_ext'] = 2 * self.v_ext * self.n_ks
        per_site_array['u'] = self.onsite_repulsion
        per_site_array['v_term'] = self.v_term_repulsion
        per_site_array['tot'] = np.sum(np.array(per_site_array[['kin', 'v_ext', 'u', 'v_term']].tolist()), axis=1)
        kinetic_contribution = np.sum(per_site_array['kin'])
        v_ext_contribution = np.sum(per_site_array['v_ext'])
        u_contribution = np.sum(self.onsite_repulsion)
        v_term_contribution = np.sum(self.v_term_repulsion)
        total_energy = kinetic_contribution + v_ext_contribution + u_contribution + v_term_contribution
        self.energy_contributions = (total_energy, kinetic_contribution, v_ext_contribution,
                                     u_contribution, v_term_contribution)
        self.per_site_energy = per_site_array
        if not silent:
            print(f'\n{"site":30s}{" ".join([f"{i:9d}" for i in range(self.Ns)])}{"total":>12s}\n{"Kinetic energy":30s}'
                  f'{" ".join([f"{i:9.4f}" for i in self.kinetic_contributions])}{kinetic_contribution:12.7f}\n'
                  f'{"External potential energy":30s}{" ".join([f"{i:9.4f}" for i in per_site_array["v_ext"]])}'
                  f'{v_ext_contribution:12.7f}\n{"On-site repulsion":30s}'
                  f'{" ".join([f"{i:9.4f}" for i in self.onsite_repulsion])}{u_contribution:12.7f}\n'
                  f'{"V-term repulsion":30s}{" ".join([f"{i:9.4f}" for i in self.v_term_repulsion])}{v_term_contribution:12.7f}\n'
                  f'{"Occupations":30s}{" ".join([f"{i:9.4f}" for i in self.n_ks * 2])}{np.sum(self.n_ks) * 2:12.7f}')
            print(f'{"_" * 20}\nTotal energy:{total_energy}')

        return total_energy

    def compare_densities_FCI(self, pass_object: Union[bool, class_Quant_NBody.QuantNBody] = False,
                              calculate_per_site=False):
        if type(pass_object) != bool:
            mol_full = pass_object
        else:
            mol_full = class_Quant_NBody.QuantNBody(self.Ns, self.Ne)
            mol_full.build_operator_a_dagger_a()
        mol_full.build_hamiltonian_fermi_hubbard(self.t + np.diag(self.v_ext), self.u4d)
        mol_full.diagonalize_hamiltonian()
        y_ab = mol_full.calculate_1rdm()
        densities = y_ab.diagonal()
        kinetic_contribution = np.sum(y_ab * self.t) * 2
        v_ext_contribution = np.sum(2 * self.v_ext * densities)
        total_energy = mol_full.ei_val[0]
        u_contribution = total_energy - kinetic_contribution - v_ext_contribution

        if calculate_per_site:
            per_site_array = np.zeros(self.Ns, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),
                                                      ('v_term', float)])
            on_site_repulsion_array = self.u4d * mol_full.calculate_2rdm_fh(index=0)
            t_multiplied_matrix = y_ab * self.t * 2
            for site in range(self.Ns):
                per_site_array[site]['v_ext'] = 2 * self.v_ext[site] * densities[site]
                per_site_array[site]['u'] = on_site_repulsion_array[site, site, site, site]
                per_site_array[site]['kin'] = np.sum(t_multiplied_matrix[site])
                per_site_array[site]['v_term'] = 0.5 * (np.sum(on_site_repulsion_array[site, site, :, :]) +
                                                        np.sum(on_site_repulsion_array[:, :, site, site])) - \
                                                 on_site_repulsion_array[site, site, site, site]
                per_site_array[site]['tot'] = sum(per_site_array[site])
            u_contribution = np.sum(per_site_array['u'])
            v_term_contribution = np.sum(per_site_array['v_term'])

            return y_ab, mol_full, (total_energy, kinetic_contribution, v_ext_contribution,
                                    u_contribution, v_term_contribution), per_site_array
        print("FCI densities (per spin):", densities)
        print(f'FCI energy: {mol_full.ei_val[0]}')
        return y_ab, mol_full, (total_energy, kinetic_contribution, v_ext_contribution,
                                u_contribution, 0)

    def plot_density_evolution(self):
        self.density_progress = np.array(self.density_progress)
        for i in range(self.density_progress.shape[1]):
            plt.plot(self.density_progress[:, i])
        plt.xlabel("Iteration")
        plt.ylabel("Density")
        plt.title("Evolution of density in simulation")
        plt.show()

    def log_casci(self, site_id, y_a_correct_imp, P, v, h_tilde, h_tilde_dimer):
        self.report_string += f'\t\t\tNew 1RDM that is optained by replacing indices 0 and {site_id}\n\t\t\t'
        temp1 = print_matrix(y_a_correct_imp, ret=True).replace('\n', '\n\t\t\t')[:-3]
        self.report_string += temp1

        self.report_string += "\t\t\tv vector\n\t\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in v[:, 0]]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"

        self.report_string += f'\t\t\tP matrix\n\t\t\t'
        temp1 = print_matrix(P, ret=True).replace('\n', '\n\t\t\t')[:-3]
        self.report_string += temp1

        self.report_string += f'\t\t\th tilde\n\t\t\t'
        temp1 = print_matrix(h_tilde, ret=True).replace('\n', '\n\t\t\t')[:-3]
        self.report_string += temp1

        self.report_string += f'\t\t\th tilde dimer\n\t\t\t'
        temp1 = print_matrix(h_tilde_dimer, ret=True).replace('\n', '\n\t\t\t')[:-3]
        self.report_string += temp1

        self.report_string += f'\t\t\tU0 parameter: {self.u[site_id]}\n'
        self.report_string += f'\t\t\tStarting impurity chemical potential mu_imp: {-self.v_hxc[site_id]}\n'

    def log_scl(self, old_density, mean_square_difference_density, i, tolerance, num_iter):
        self.report_string += f"\tNew densities: "
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.n_ks]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"
        self.report_string += f"\tOld densities: "
        if type(old_density) == float:
            temp1 = ["inf"]
        else:
            temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in old_density]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"
        self.report_string += f"\taverage square difference: {mean_square_difference_density}\n\tStopping" \
                              f" condtition: ({mean_square_difference_density:8.4f}<{tolerance:8.4f}) OR " \
                              f"({i + 1}>={num_iter})\n"

    def plot_hubbard_molecule(self):
        G = nx.Graph()
        colors = ['lightgrey', 'mistyrose', 'lightcyan', 'thistle', 'springgreen', 'yellow', 'cyan', 'magenta',
                  'orange']
        color_map = []
        labeldict = {}
        edge_labels = dict()
        for i in range(self.Ns):
            node_string = f"U={self.u[i]:.1f}\nv_ext={self.v_ext[i]:.3f}"
            for key, value in self.equiv_atom_groups.items():
                if i in value:
                    group_id = key
                    break
            else:
                group_id = -1
            G.add_nodes_from([(i, {'color': colors[group_id]})])
            labeldict[i] = node_string
            for j in range(i, self.Ns):
                if self.t[i, j] != 0:
                    edge_string = f"{self.t[i, j]}"
                    G.add_edge(i, j)
                    edge_labels[(i, j)] = edge_string
        for i in G.nodes:
            color_map.append(G.nodes[i]['color'])
        fig, ax = plt.subplots(1, 1)
        position = nx.spring_layout(G)
        nx.draw(G, pos=position, ax=ax, labels=labeldict, with_labels=True, node_color=color_map, node_size=5000,
                font_weight='bold')
        nx.draw_networkx_edge_labels(G, position, edge_labels)
        ax.set_xlim(*np.array(ax.get_xlim()) * 1.3)
        ax.set_ylim(*np.array(ax.get_ylim()) * 1.3)
        print(f"results/{self.description}_molecule.png")
        fig.show()
        fig.savefig(f"results/{self.description}_molecule.svg")

    def clear_object(self, description=''):
        """
        This method is used to clear all data that would mess with the self consistent loop without reloading
        a_dagger_a. This enables us faster calculations
        :param description:
        :return: Nada
        """
        self.description = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}" + description
        self.u = np.array((), dtype=np.float64)
        self.t = np.array((), dtype=np.float64)
        self.v_ext = np.array((), dtype=np.float64)
        self.equiv_atom_groups = dict()
        self.v_hxc = np.zeros(self.Ns, dtype=np.float64)  # Hartree exchange correlation potential
        self.report_string += f'\n{"#" * 50}\nClear of data! From now on There is a new object \n' \
                              f'{"#" * 50}\n\nObject (still) with {self.Ns} sites and {self.Ne} electrons\n'
        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.v_hxc_progress = []

        self.oscillation_correction_dict = dict()


def cost_function_whole(v_hxc_approximation: np.array, mol_obj: Molecule) -> np.array:
    mol_obj.v_hxc = np.zeros(mol_obj.Ns, float)
    for group_id, group_site_tuple in mol_obj.equiv_atom_groups.items():
        if group_id != 0:
            for site_id_i in group_site_tuple:
                mol_obj.v_hxc[site_id_i] = v_hxc_approximation[group_id - 1]
    mol_obj.v_hxc_progress.append(mol_obj.v_hxc)
    mol_obj.calculate_ks()
    mol_obj.density_progress.append(mol_obj.n_ks)
    output_array = np.zeros(len(mol_obj.equiv_atom_groups) - 1, float)
    mu_imp_first = np.nan
    first_iteration = True
    output_index = 0
    for site_group in mol_obj.equiv_atom_groups.keys():
        site_id = mol_obj.equiv_atom_groups[site_group][0]
        y_a_correct_imp = change_indices(mol_obj.y_a, site_id)
        P, v = Quant_NBody.householder_transformation(y_a_correct_imp)
        h_tilde = P @ (change_indices(mol_obj.t, site_id) + np.diag(change_indices(mol_obj.v_s, site_id))) @ P

        h_tilde_dimer = h_tilde[:2, :2]
        if mol_obj.v_term:
            u4d = np.zeros(mol_obj.u4d.shape)
            u4d[site_id, site_id, :, :] += mol_obj.u4d[site_id, site_id, :, :] / 2
            u4d[:, :, site_id, site_id] += mol_obj.u4d[:, :, site_id, site_id] / 2
            if site_id == 0:
                u4d_correct_indices = u4d
            else:
                mask = np.arange(mol_obj.Ns)
                mask[0] = site_id
                mask[site_id] = 0
                u4d_correct_indices = u4d[:, :, :, mask][:, :, mask, :][:, mask, :, :][mask, :, :, :]
            u_0_dimer = np.einsum('ap, bq, cr, ds, abcd -> pqrs', P, P, P, P, u4d_correct_indices)[:2, :2, :2, :2]
        else:
            u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
            u_0_dimer[0, 0, 0, 0] += mol_obj.u[site_id]

        mol_obj.log_casci(site_id, y_a_correct_imp, P, v, h_tilde, h_tilde_dimer)
        mol_obj.h_tilde_dimer[site_group] = h_tilde_dimer

        if first_iteration:
            if 0 not in mol_obj.equiv_atom_groups[site_group]:
                raise Exception("Unexpected behaviour: First impurity site should have been the 0th site")
            sol = sc_opt.root_scalar(cost_function_casci_root,
                                     args=(mol_obj.embedded_mol, h_tilde_dimer, u_0_dimer, mol_obj.n_ks[site_id]),
                                     bracket=[-5, 15], method='brentq', options={'xtol': 1e-6})
            mu_imp, function_calls = sol.root, sol.function_calls
            mu_imp_first = mu_imp
        else:
            mu_imp = mu_imp_first + mol_obj.v_hxc[site_id]
            error_i = cost_function_casci_root(mu_imp, mol_obj.embedded_mol, h_tilde_dimer, u_0_dimer,
                                               mol_obj.n_ks[site_id])
            output_array[output_index] = error_i
            output_index += 1
        two_rdm = mol_obj.embedded_mol.calculate_2rdm_fh(index=0)
        on_site_repulsion_i = two_rdm[0, 0, 0, 0] * u_0_dimer[0, 0, 0, 0]
        v_term_repulsion_i = np.sum(two_rdm * u_0_dimer) - on_site_repulsion_i
        for every_site_id in mol_obj.equiv_atom_groups[site_group]:
            mol_obj.kinetic_contributions[every_site_id] = 2 * h_tilde[1, 0] * mol_obj.embedded_mol.one_rdm[1, 0]
            mol_obj.onsite_repulsion[every_site_id] = on_site_repulsion_i
            mol_obj.imp_potential[every_site_id] = mu_imp
            mol_obj.v_term_repulsion[every_site_id] = v_term_repulsion_i
        first_iteration = False
    rms = np.sqrt(np.mean(np.square(output_array)))
    # print(f"for input {''.join(['{num:{dec}}'.format(num=cell, dec='+10.4f') for cell in mol_obj.v_hxc])} we get error"
    #       f" {''.join(['{num:{dec}}'.format(num=cell, dec='+10.4f') for cell in output_array])} "
    #       f" (RMS = {rms})")
    print(
        f"for input {''.join(['{num:{dec}}'.format(num=cell, dec='+13.3e') for cell in mol_obj.v_hxc[:2]])} we get error"
        f" {''.join(['{num:{dec}}'.format(num=cell, dec='+13.3e') for cell in output_array[:2]])} "
        f" (RMS = {rms})")
    # if rms < 1e-4:
    #     return np.zeros(len(mol_obj.equiv_atom_groups) - 1, float)
    return output_array


def cost_function_casci_root(mu_imp, embedded_mol, h_tilde_dimer, u_0_dimer, desired_density):
    # mu_imp = mu_imp[0]
    global ITERATION_NUM
    ITERATION_NUM += 1
    mu_imp_array = np.array([[mu_imp, 0], [0, 0]])
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer - mu_imp_array, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()

    density_dimer = embedded_mol.calculate_1rdm(index=0)
    return density_dimer[0, 0] - desired_density


def see_landscape_ruggedness(embedded_mol, h_tilde_dimer, u_0_dimer, goal_density=False, optimized_potential=False,
                             num_dim=1, arange=(-2, 2 + 0.1, 0.1)):
    if num_dim == 1:
        x = np.arange(*arange)
        y = []
        for mu_imp in x:
            abs_error = np.sqrt(cost_function_casci([mu_imp], embedded_mol, h_tilde_dimer, u_0_dimer, 0))
            y.append(abs_error)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y)
        if goal_density:
            plt.hlines(goal_density, *ax.get_xlim(), label="Goal density")
        if optimized_potential:
            plt.vlines(optimized_potential, *ax.get_ylim(), label='return of optimizer')
        string1 = f"{datetime.now().strftime('%d_%H_%M_%S_%f')}"
        ax.set_title(string1)
        fig.savefig("results/" + string1 + ".png")


def generate_from_graph(sites, connections):
    """
    We can provide graph information and program generates hamiltonian automatically
    :param sites: in the type of: {0:{'v':0, 'U':4}, 1:{'v':1, 'U':4}, 2:{'v':0, 'U':4}, 3:{'v':1, 'U':4}}
    :param connections: {(0, 1):1, (1, 2):1, (2, 3):1, (0,3):1}
    :return: h and U parameters
    """
    n_sites = len(sites)
    t = np.zeros((n_sites, n_sites), dtype=np.float64)
    v = np.zeros(n_sites, dtype=np.float64)
    u = np.zeros(n_sites, dtype=np.float64)
    for id, params in sites.items():
        if 'U' in params:
            u[id] = params['U']
        elif 'u' in params:
            u[id] = params['u']
        else:
            raise "Problem with params: " + params
        v[id] = params['v']
    for pair, param in connections.items():
        t[pair[0], pair[1]] = -param
        t[pair[1], pair[0]] = -param
    return t, v, u


if __name__ == "__main__":
    i = 2
    U_ = 1
    name = 'chain1'
    mol1 = Molecule(6, 6, name)
    mol_full = class_Quant_NBody.QuantNBody(6, 6)
    mol_full.build_operator_a_dagger_a()
    first = False
    pmv = i
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = []
    for j in range(6):
        nodes_dict[j] = {'v': (j - 2.5) * i, 'U': 10}
        if j != 6 - 1:
            edges_dict[(j, j + 1)] = 1
        eq_list.append([j])
    t, v_ext, u = generate_from_graph(nodes_dict, edges_dict)
    mol1.add_parameters(u, t, v_ext, eq_list, 0.5)
    end_str = '\n'
    print(mol1.v_ext)
    mol1.v_hxc = np.zeros(mol1.Ns)
    mol1.v_hxc_progress = []
    mol1.density_progress = []
    start1 = datetime.now()
    # mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
    mol1.find_solution_as_root()
    end1 = datetime.now()
    end_str += str(end1 - start1) + '\n'
    mol1.compare_densities_FCI(False, True)
    print(end_str, ITERATION_NUM)
