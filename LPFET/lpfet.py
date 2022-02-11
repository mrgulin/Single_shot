import class_Quant_NBody
import Quant_NBody
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc_opt
from datetime import datetime
from essentials import OUTPUT_SEPARATOR, OUTPUT_FORMATTING_NUMBER, print_matrix, generate_1rdm

# For plotting of the molecule (if you don't need this you can delete Molecule.plot_hubbard_molecule
# and this import statements
import pandas as pd
import networkx as nx


def log_calculate_ks_decorator(func):
    """
    Wrapper that adds data to Molecule.report_string for method calculate_ks. Doesn't affect behaviour of the program!!
    :param func:  calculate_ks function
    :return: nested function
    """

    def wrapper_func(self):
        ret_val = func(self)

        self.report_string += f"\tEntered calculate_ks\n"

        self.report_string += f"\t\tHartree exchange correlation chemical potential\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.mu_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + '\n'

        self.report_string += "\t\tmu_s\n\t\t"
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.mu_s]
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
    def wrapper_func(self, u, t, v_ext, equiv_atom_group_list):
        ret_val = func(self, u, t, v_ext, equiv_atom_group_list)
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
        self.t = np.array((), dtype=np.float64)  # 2D one electron Hamiltonian with all terms for i != j
        self.v_ext = np.array((), dtype=np.float64)  # 1D one electron Hamiltonian with all terms for i == j
        self.equiv_atom_groups = dict()

        # density matrix
        self.y_a = np.array((), dtype=np.float64)  # y --> gamma, a --> alpha so this indicates it is only per one spin

        # KS
        self.mu_s = np.zeros(self.Ns, dtype=np.float64)  # Kohn-Sham potential
        self.h_ks = np.array((), dtype=np.float64)
        self.wf_ks = np.array((), dtype=np.float64)  # Kohn-Sham wavefunciton
        self.epsilon_s = np.array((), dtype=np.float64)  # Kohn-Sham energies
        self.n_ks = np.array((), dtype=np.float64)  # Densities of KS sysyem
        self.mu_hxc = np.zeros(self.Ns, dtype=np.float64)  # Hartree exchange correlation potential

        # Quant_NBody objects
        # self.whole_mol = class_Quant_NBody.QuantNBody(self.Ns, self.Ne)
        self.embedded_mol = class_Quant_NBody.QuantNBody(2, 2)
        self.embedded_mol.build_operator_a_dagger_a()

        self.report_string = f'Object with {self.Ns} sites and {self.Ne} electrons\n'

        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.mu_hxc_progress = []

    @log_add_parameters_decorator
    def add_parameters(self, u, t, v_ext, equiv_atom_group_list):
        if len(u) != self.Ns or len(t) != self.Ns or len(v_ext) != self.Ns:
            raise f"Problem with size of matrices: U={len(u)}, t={len(t)}, v_ext={len(v_ext)}"
        self.u = u
        self.t = t
        self.v_ext = v_ext
        for index, item in enumerate(equiv_atom_group_list):
            self.equiv_atom_groups[index] = tuple(item)

    def self_consistent_loop(self, num_iter=10, tolerance=0.0001, overwrite_output="", oscillation_compensation=0):
        self.report_string += "self_consistent_loop:\n"
        old_density = np.inf
        i = 0
        for i in range(num_iter):
            self.report_string += f"Iteration # = {i}\n"
            self.calculate_ks()
            self.density_progress.append(self.n_ks.copy())
            self.CASCI(oscillation_compensation)
            self.mu_hxc_progress.append(self.mu_hxc.copy())
            print(f"Loop {i}", end = ', ')
            mean_square_difference_density = np.average(np.square(self.n_ks - old_density))

            self.log_scl(old_density, mean_square_difference_density, i, tolerance, num_iter)

            if mean_square_difference_density < tolerance:
                break
            old_density = self.n_ks
        self.report_string += f'Final Hxc chemical potential:\n'
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.mu_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"
        if overwrite_output:
            conn = open(overwrite_output, "w", encoding="UTF-8")
        else:
            conn = open(f"results/{self.description}_log.dat", "w", encoding="UTF-8")
        conn.write(self.report_string)
        conn.close()
        return i

    @log_calculate_ks_decorator
    def calculate_ks(self):
        self.mu_s = self.mu_hxc - self.v_ext
        self.h_ks = self.t - np.diag(self.mu_s)
        self.epsilon_s, self.wf_ks = np.linalg.eigh(self.h_ks, 'U')
        self.y_a = generate_1rdm(self.Ns, self.Ne, self.wf_ks)
        self.n_ks = np.copy(self.y_a.diagonal())

    def CASCI(self, oscillation_compensation=0):
        self.report_string += "\tEntered CASCI\n"
        for site_group in self.equiv_atom_groups.keys():
            self.report_string += f"\t\tGroup {site_group} with sites {self.equiv_atom_groups[site_group]}\n"
            site_id = self.equiv_atom_groups[site_group][0]

            # Householder transforms impurity on index 0 so we have to make sure that impurity is on index 0:
            if site_id == 0:
                y_a_correct_imp = np.copy(self.y_a)
            else:
                # We have to move impurity on the index 0
                y_a_correct_imp = np.copy(self.y_a)
                y_a_correct_imp[:, [0, site_id]] = y_a_correct_imp[:, [site_id, 0]]
                y_a_correct_imp[[0, site_id], :] = y_a_correct_imp[[site_id, 0], :]

            P, v = Quant_NBody.Householder_transformation(y_a_correct_imp)
            h_tilde = P @ (self.t + np.diag(self.v_ext)) @ P
            # TODO: Ask Fromager how should this be!!!

            h_tilde_dimer = h_tilde[:2, :2]
            u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
            u_0_dimer[0, 0, 0, 0] += self.u[site_id]
            # h_tilde_dimer[0,0] += self.v_ext[site_id]
            mu_imp = self.mu_hxc[[site_id]]  # Double parenthesis so I keep array, in future this will be list of
            # indices for block householder

            self.log_CASCI(site_id, y_a_correct_imp, P, v, h_tilde, h_tilde_dimer)

            opt_v_imp_obj = sc_opt.minimize(cost_function_CASCI, mu_imp,
                                            args=(self.embedded_mol, h_tilde_dimer, u_0_dimer, self.n_ks[site_id]),
                                            method='BFGS', tol=1e-2)
            # This minimize cost function (difference between KS occupations and CASCI occupations squared)
            error = opt_v_imp_obj['fun']
            mu_imp = opt_v_imp_obj['x'][0]

            # see_landscape_ruggedness(self.embedded_mol, h_tilde_dimer, u_0_dimer, goal_density=self.n_ks[site_id],
            #                          optimized_potential=mu_imp, num_dim=1)

            self.report_string += f'\t\t\tOptimized chemical potential mu_imp: {mu_imp}\n'
            self.report_string += f'\t\t\tError in densities (square): {error}\n'

            # print(f"managed to get E^2={error} with mu_imp={mu_imp}")

            self.update_mu_hxc(site_group, mu_imp, oscillation_compensation)

        self.report_string += f'\t\tHxc chemical potential in the end of a cycle:\n\t\t'
        temp1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in self.mu_hxc]
        self.report_string += f'{OUTPUT_SEPARATOR}'.join(temp1) + "\n"

    def update_mu_hxc(self, site_group, mu_imp, oscillation_compensation):
        if oscillation_compensation == 1:
            if len(self.mu_hxc_progress) < 2:
                oscillation_compensation = 0
            else:
                index = self.equiv_atom_groups[site_group][0]
                mu_minus_2 = self.mu_hxc_progress[-2][index]
                mu_minus_1 = self.mu_hxc_progress[-1][index]
                if (mu_minus_2 - mu_minus_1) * (mu_minus_1 - mu_imp) < 0 and\
                        abs(mu_minus_2 - mu_minus_1) * 0.75 < abs(mu_minus_1 - mu_imp):
                    # First statement means that potential correction turned direction and second means that it is large
                    new_mu_imp = mu_minus_1 + (mu_imp - mu_minus_1) * 0.75
                    print(f'{mu_minus_2}->{mu_minus_1}->{new_mu_imp}!={mu_imp}')
                    mu_imp = new_mu_imp
                for every_site_id in self.equiv_atom_groups[site_group]:
                    self.mu_hxc[every_site_id] = mu_imp
        if oscillation_compensation == 0:
            for every_site_id in self.equiv_atom_groups[site_group]:
                self.mu_hxc[every_site_id] = mu_imp


    def compare_densities_FCI(self, pass_object=False):
        if type(pass_object) != bool:
            mol_full = pass_object
        else:
            mol_full = class_Quant_NBody.QuantNBody(self.Ns, self.Ne)
            mol_full.build_operator_a_dagger_a()
        u_4d = np.zeros((self.Ns, self.Ns, self.Ns, self.Ns))
        for i in range(self.Ns):
            u_4d[i, i, i, i] = self.u[i]
        mol_full.build_hamiltonian_fermi_hubbard(self.t + np.diag(self.v_ext), u_4d)
        mol_full.diagonalize_hamiltonian()
        y_ab = mol_full.calculate_1RDM_tot()
        print("FCI densities (per spin):", y_ab.diagonal() / 2)
        return y_ab, mol_full

    def plot_density_evolution(self):
        self.density_progress = np.array(self.density_progress)
        for i in range(self.density_progress.shape[1]):
            plt.plot(self.density_progress[:, i])
        plt.xlabel("Iteration")
        plt.ylabel("Density")
        plt.title("Evolution of density in simulation")
        plt.show()

    def log_CASCI(self, site_id, y_a_correct_imp, P, v, h_tilde, h_tilde_dimer):
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
        self.report_string += f'\t\t\tStarting impurity chemical potential mu_imp: {self.mu_hxc[site_id]}\n'

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
        colors = ['lightgrey', 'mistyrose', 'lightcyan', 'thistle', 'orange']
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
        self.mu_hxc = np.zeros(self.Ns, dtype=np.float64)  # Hartree exchange correlation potential
        self.report_string += f'\n{"#" * 50 }\nClear of data! From now on There is a new object \n' \
                              f'{"#" * 50 }\n\nObject (still) with {self.Ns} sites and {self.Ne} electrons\n'
        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.mu_hxc_progress = []

def cost_function_CASCI(mu_imp, embedded_mol, h_tilde_dimer, u_0_dimer, desired_density):
    mu_imp = mu_imp[0]
    mu_imp_array = np.array([[mu_imp, 0], [0, 0]])
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer + mu_imp_array, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()

    density_dimer = Quant_NBody.Build_One_RDM_alpha(embedded_mol.WFT_0, embedded_mol.a_dagger_a)
    return (density_dimer[0, 0] - desired_density) ** 2

def see_landscape_ruggedness(embedded_mol, h_tilde_dimer, u_0_dimer, goal_density=False, optimized_potential=False,
                             num_dim=1, arange=(-2, 2 + 0.1 , 0.1)):
    if num_dim == 1:
        x = np.arange(*arange)
        y = []
        for mu_imp in x:
            abs_error = np.sqrt(cost_function_CASCI([mu_imp], embedded_mol, h_tilde_dimer, u_0_dimer, 0))
            y.append(abs_error)
        fig, ax = plt.subplots(1,1)
        ax.plot(x, y)
        if goal_density:
            plt.hlines(goal_density, *ax.get_xlim(), label="Goal density")
        if optimized_potential:
            plt.vlines(optimized_potential, *ax.get_ylim(), label='return of optimizer')
        string1 = f"{datetime.now().strftime('%d_%H_%M_%S_%f')}"
        ax.set_title(string1)
        fig.savefig("results/" + string1+".png")

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
    pmv = 0.1
    mol1 = Molecule(6, 6, f'ring6_2sites_{pmv}')

    t, v_ext, u = generate_from_graph(
        {0: {'v': -pmv, 'U': 1}, 1: {'v': pmv, 'U': 1}, 2: {'v': pmv, 'U': 1}, 3: {'v': -pmv, 'U': 1},
         4: {'v': pmv, 'U': 1}, 5: {'v': pmv, 'U': 1}},
        {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (0, 5): 1})
    mol1.add_parameters(u, t, v_ext, [[0, 3], [1, 2, 4, 5]])
    mol1.clear_object("New object :)")
    mol1.add_parameters(u, t, v_ext, [[0, 3], [1, 2, 4, 5]])
    # mol1.plot_hubbard_molecule()
    # mol1.self_consistent_loop()
    # y_real, mol_full = mol1.compare_densities_FCI()
