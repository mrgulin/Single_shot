from LPFET import essentials
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from datetime import datetime
# For plotting of the molecule (if you don't need this you can delete Molecule.plot_hubbard_molecule
# and this import statements
import networkx as nx
from sklearn.linear_model import LinearRegression
from typing import Union
import typing
import quantnbody as qnb  # Folder Quant_NBody has to be in the sys.path or installed as package.
import quantnbody_class_new as class_qnb
from essentials import generate_1rdm
from LPFET import errors
import logging

# logging.basicConfig(filename='optimizer-comparison.log', level=logging.DEBUG)
formatter = logging.Formatter('%(levelname)8s %(lineno)4s %(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
general_handler = logging.FileHandler('general_lpfet.log', mode='w')
general_handler.setFormatter(formatter)
general_handler.setLevel(20)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(25)

general_logger = logging.getLogger('lpfet_general_logger')
general_logger.setLevel(-1)
general_logger.addHandler(general_handler)
general_logger.addHandler(stream_handler)

# optimize_handler = logging.FileHandler('optimize_root.log', mode='w')
# optimize_handler.setFormatter(formatter)
# optimize_logger = logging.getLogger('optimizing_root')
# optimize_logger.setLevel(logging.DEBUG)
# optimize_logger.addHandler(optimize_handler)
# optimize_logger.addHandler(stream_handler)

COMPENSATION_1_RATIO = 0.5  # for the Molecule.update_v_hxc
COMPENSATION_MAX_ITER_HISTORY = 4
COMPENSATION_5_FACTOR = 1
COMPENSATION_5_FACTOR2 = 0.5
ITERATION_NUM = 0
ROOT_FINDING_LIST_INPUT = []
ROOT_FINDING_LIST_OUTPUT = []

ROOT_LPFET_SOLVER_MAX_ITER = 100
np.seterr(all='raise')
np.errstate(all='raise')
np.set_printoptions(linewidth=np.inf)


def abs_norm(x):
    return np.max(np.abs(x))


def change_indices(array_inp: np.array, site_id: typing.Union[int, typing.List[int]],
                   to_index: typing.Union[int, typing.List[int], None] = None):
    array = np.copy(array_inp)
    if isinstance(site_id, (int, np.int64, np.int32, float, np.float64)):
        site_id = [int(site_id)]
        if to_index is not None:
            to_index = [int(to_index)]
        else:
            to_index = [0]
    else:
        site_id = [int(i) for i in site_id]

        if to_index is not None:
            to_index = [int(i) for i in to_index]
        else:
            to_index = list(range(len(site_id)))
        # It mustn't be a numpy array because we want + operator to be concatenation and not sum of arrays
    if site_id != to_index:
        set_list = to_index + site_id
        get_list = site_id + to_index
        if len(np.unique(set_list)) != len(set_list):
            # In this case simple approach don't work since we are writing and reading from same column. We have to get
            # rid of the duplicate enteries! we do this by next formula:
            set_list2 = []
            get_list2 = []
            buffer_set = 0
            buffer_get = 0
            set_list = to_index + site_id
            get_list = site_id + to_index
            for i in range(len(set_list)):
                while i + buffer_set < len(set_list) and set_list[i + buffer_set] in set_list2:
                    buffer_set += 1
                while i + buffer_get < len(get_list) and get_list[i + buffer_get] in get_list2:
                    buffer_get += 1
                if i + buffer_set == len(set_list) or i + buffer_get == len(get_list):
                    break
                set_list2.append(set_list[buffer_set + i])
                get_list2.append(get_list[buffer_get + i])
            # general_logger.error(f"Special case with column switching: "
            #                      f"{get_list}, {set_list} --> {get_list2}, {set_list2}")
            get_list = get_list2
            set_list = set_list2
        # We have to move impurity on the index 0
        if array_inp.ndim == 2:
            array[:, set_list] = array[:, get_list]
            array[set_list, :] = array[get_list, :]
        elif array_inp.ndim == 1:
            array[set_list] = array[get_list]
    return array


def normalize_values(starting_vals):
    # starting_vals = [0, 10000, -312, 10000, 0, 10000]
    starting_vals2 = np.zeros(len(starting_vals), int) + 1
    arg_sort = np.argsort(starting_vals)
    for i in range(1, len(arg_sort)):
        if not starting_vals[arg_sort[i]] > starting_vals[arg_sort[i - 1]] + 1e-4:
            starting_vals2[i] = starting_vals2[i - 1]
        else:
            starting_vals2[i] = starting_vals2[i - 1] + 1
    for i in range(len(starting_vals)):
        starting_vals[arg_sort[i]] = starting_vals2[i]
    return np.array(starting_vals)


def cangen(t, cannon_list):
    order = normalize_values(cannon_list)
    order_old = np.zeros(len(order), int) - 100
    iter1 = 0
    while iter1 < 15 or sum(order - order_old) != 0:

        order_new = np.zeros(len(order), int)
        for i in range(len(cannon_list)):
            order_new[i] = order[i] * 100 + np.sum(t[i] * order)
        iter1 += 1
        order_old = order.copy()
        order = normalize_values(order_new)
    ret_list = []
    for value in order:
        new_val = tuple(np.where(order == value)[0])
        if new_val not in ret_list:
            ret_list.append(new_val)
    return ret_list


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
        self.equiv_atom_groups = []
        self.equiv_atom_groups_reverse = []
        self.v_term = None

        # density matrix
        self.y_a = np.array((), dtype=np.float64)  # y --> gamma, a --> alpha so this indicates it is only per one spin

        # KS
        self.v_s = np.zeros(self.Ns, dtype=np.float64)  # Kohn-Sham potential
        self.h_ks = np.array((), dtype=np.float64)
        self.wf_ks = np.array((), dtype=np.float64)  # Kohn-Sham wave function
        self.epsilon_s = np.array((), dtype=np.float64)  # Kohn-Sham energies
        self.n_ks = np.array((), dtype=np.float64)  # Densities of KS system
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
        # self.whole_mol = class_qnb.QuantNBody(self.Ns, self.Ne)
        self.embedded_mol = class_qnb.HamiltonianV2(2, 2)
        self.embedded_mol_dict = dict()
        self.embedded_mol.build_operator_a_dagger_a(silent=True)

        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.v_hxc_progress = []

        self.iteration_i = 0
        self.oscillation_correction_dict = dict()

        self.h_tilde_dimer = dict()

        self.compensation_ratio_dict = dict()

        # Block Householder
        self.blocks = []  # This 2d list tells us which sites are merged into which clusters [[c0s0, c0s1,..],[c1s0...]]
        self.equiv_block_groups = []  # This 2d list tells us which blocks are equivalent. [0, 1], [2]] means that first
        # block is same as second and 3rd is different
        self.equiv_atoms_in_block = []  # This 2d list is important so we know which atoms have the same impurity
        # potentials. This list is important because we have to optimize len(..) - 1 Hxc potentials
        self.block_hh = False  # Variable that will be checked to see if algorithm should do single impurity or not.

        self.optimize_progress_input = []
        self.optimize_progress_output = []

    def add_parameters(self, u, t, v_ext,
                       v_term_repulsion_ratio: typing.Union[bool, float] = False):
        if len(u) != self.Ns or len(t) != self.Ns or len(v_ext) != self.Ns:
            raise Exception(f"Problem with size of matrices: U={len(u)}, t={len(t)}, v_ext={len(v_ext)}")
        self.u = u
        self.t = t
        self.v_ext = v_ext
        for i in range(self.Ns):
            if v_term_repulsion_ratio:
                self.v_term = np.zeros((self.Ns, self.Ns))
                for a in range(len(self.t)):
                    for b in range(len(self.t)):
                        if self.t[a, b] != 0 and a > b:
                            self.v_term[a, b] = v_term_repulsion_ratio * 0.5 * (u[a] + u[b])
                            # self.v_term[b, a] = v_term_repulsion_ratio * 0.5 * (u[a] + u[b])
        if not np.allclose(self.t, self.t.T):
            raise Exception("t matrix should have been symmetric")
        equiv_atom_group_list = cangen(t, np.array(u) * 100 + np.array(v_ext))
        general_logger.info(f'Equivalent atoms: {equiv_atom_group_list}')
        self.equiv_atom_groups = []
        for index, item in enumerate(equiv_atom_group_list):
            self.equiv_atom_groups.append(tuple(item))
            self.compensation_ratio_dict[index] = COMPENSATION_5_FACTOR
        self.equiv_atom_groups_reverse = np.zeros(self.Ns, int)
        for group_id, atom_list in enumerate(self.equiv_atom_groups):
            for site_id in atom_list:
                self.equiv_atom_groups_reverse[site_id] = group_id

    def prepare_for_block(self, blocks: typing.List[typing.List[int]]):
        n_blocks = len(blocks)
        if 0 not in blocks[0]:
            general_logger.error('site 0 must be in the first block')
            return 0
        self.blocks = blocks
        normalized_blocks = [[int(self.equiv_atom_groups_reverse[j]) for j in i] for i in blocks]
        ignored = []
        equiv_block_list = []
        for i in range(n_blocks):
            if i in ignored:
                continue
            equiv_block_list.append([i])
            for j in range(i + 1, n_blocks):
                if i in ignored:
                    continue
                if sorted(normalized_blocks[i]) == sorted(normalized_blocks[j]):
                    equiv_block_list[-1].append(j)
                    ignored.append(j)
        self.equiv_block_groups = equiv_block_list
        ignore = []
        equiv_atoms_in_block = []
        for i in range(self.Ns):
            if i in ignore:
                continue
            atom_group = int(self.equiv_atom_groups_reverse[i])
            equivalent_atoms = self.equiv_atom_groups[atom_group]
            block_i = [i in h for h in self.blocks].index(True)
            block_group_i = [block_i in h for h in self.equiv_block_groups].index(True)
            equiv_atoms_in_block.append([i])
            for j in equivalent_atoms:
                if i == j or j in ignore:
                    continue
                    # Only if i!=j and if i>j so we don't count things twice
                block_j = [j in h for h in self.blocks].index(True)
                block_group_j = [block_j in h for h in self.equiv_block_groups].index(True)
                if block_group_j == block_group_i:
                    equiv_atoms_in_block[-1].append(j)
                    ignore.append(j)
            ignore.append(i)
        general_logger.info(f"Eqivalent blocks: {self.equiv_block_groups}")
        general_logger.info(f"Equvialent atoms inside blocks: {equiv_atoms_in_block}")
        self.equiv_atoms_in_block = equiv_atoms_in_block

        for block_i in blocks:
            block_size = len(block_i)
            self.embedded_mol_dict[block_size] = class_qnb.HamiltonianV2(block_size * 2, block_size * 2)
            self.embedded_mol_dict[block_size].build_operator_a_dagger_a(silent=True)
        self.block_hh = True

    def self_consistent_loop(self, num_iter=10, tolerance=0.0001,
                             oscillation_compensation: typing.Union[int, typing.List[int]] = 0, v_hxc_0=None):
        old_density = np.inf
        old_v_hxc = np.inf
        i = 0
        for i in range(num_iter):
            self.iteration_i = i
            self.calculate_ks()
            self.density_progress.append(self.n_ks.copy())
            self.casci(oscillation_compensation, v_hxc_0)
            self.v_hxc_progress.append(self.v_hxc.copy())
            general_logger.info(f"\nLoop {i}", end=', ')
            mean_square_difference_density = np.average(np.square(self.n_ks - old_density))
            max_difference_v_hxc = np.max(np.abs(self.v_hxc - old_v_hxc))

            if mean_square_difference_density < tolerance and max_difference_v_hxc < 0.01:
                break
            old_density = self.n_ks
            old_v_hxc = self.v_hxc.copy()
        return i

    def find_solution_as_root(self, starting_approximation=None):
        if self.block_hh:
            eq_atom = self.equiv_atoms_in_block
        else:
            eq_atom = self.equiv_atom_groups

        if starting_approximation is None:
            starting_approximation = np.zeros(len(eq_atom) - 1, float)
            for group_key, group_tuple in enumerate(eq_atom):
                if group_key != 0:
                    group_element_site = group_tuple[0]
                    starting_approximation[group_key - 1] = (self.v_ext[0] - self.v_ext[group_element_site]) * 0.5
        elif len(starting_approximation) != len(eq_atom) - 1:
            raise Exception(f'Wrong length of the starting approximation: len(starting_approximation) != '
                            f'len(eq_atom) - 1 ({len(starting_approximation)} != {len(eq_atom) - 1}')
        if self.block_hh:
            opt_function = cost_function_whole_block
        else:
            opt_function = cost_function_whole
        general_logger.log(25, f'starting to to optimization of Hxc potentials, '
                               f'starting approximation = {starting_approximation}')
        self.optimize_progress_output = []
        self.optimize_progress_input = []
        model = scipy.optimize.root(opt_function, starting_approximation,
                                    args=(self,), options={'fatol': 1e-2, "maxfev": ROOT_LPFET_SOLVER_MAX_ITER,
                                                           'fnorm': abs_norm, 'ftol': 0, "M": 30},
                                    method='df-sane')
        optimize_progress_output = np.array(self.optimize_progress_output)
        optimize_progress_input = np.array(self.optimize_progress_input)
        # optimize_logger.info('New optimization:')
        # algorithms = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden',
        #               'excitingmixing', 'krylov', 'df-sane']
        # options = {'hybr':{'eps':0.01, 'xtol': 1e-3}, 'lm': {'ftol': 1e-4, 'eps': 0.01},
        #            'broyden1': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'broyden2': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'anderson': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'linearmixing': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'diagbroyden': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'excitingmixing': {'fatol': 1e-2, "tol_norm": abs_norm},
        #            'krylov': {'fatol': 1e-2, "tol_norm": abs_norm}, 'df-sane': {'fatol': 1e-2,
        #                                                                         'fnorm': abs_norm, 'ftol': 0}}
        # for one_algorithm in algorithms:
        #     time_start = datetime.now()
        #     try:
        #         model = scipy.optimize.root(cost_function_whole_block, starting_approximation,
        #                                     args=(self,), method=one_algorithm, options=options[one_algorithm])
        #     except:
        #         model = scipy.optimize.OptimizeResult()
        #         model.x = np.nan
        #         model.fun = np.nan
        #     stop_time = datetime.now()
        #     optimize_logger.debug(f"{one_algorithm:>14s} {stop_time - time_start}: {np.sum(np.square(model.fun))}")

        #     model = scipy.optimize.root(cost_function_whole_block, starting_approximation,
        #                                         args=(self,), options={'xtol': 1e-4},
        #                                         method='hybr')
        if not model.success or np.sum(np.square(model.fun)) > 0.01:

            weights = np.sqrt(np.sum(np.square(1 / (optimize_progress_output + 1e-5)), axis=1))
            mask = weights > np.percentile(weights, 95)
            final_approximation1 = np.average(optimize_progress_input[mask], axis=0, weights=weights[mask])
            error1 = opt_function(final_approximation1, self)
            final_approximation2 = optimize_progress_input[weights.argsort()[::-1]][0]
            error2 = opt_function(final_approximation2, self)
            if abs_norm(error2) > abs_norm(error1):
                final_approximation = final_approximation1
                final_error = error1
            else:
                final_approximation = final_approximation2
                final_error = error2
            if abs_norm(final_error) < 5e-2:
                return final_approximation
            else:
                general_logger.error("Optimization function for the whole system didn't find solution")
                return False
        v_hxc = model.x
        general_logger.log(25, f"Optimized: nfev = {model.nfev}, fun = {model.fun}, x = {model.x}")
        general_logger.info(model)
        return v_hxc

    def calculate_ks(self, prevent_extreme_values=False):
        self.v_s = self.v_hxc + self.v_ext
        self.h_ks = self.t + np.diag(self.v_s)
        try:
            self.epsilon_s, self.wf_ks = np.linalg.eigh(self.h_ks, 'U')
        except np.linalg.LinAlgError as e:
            raise Exception(f"calculate_ks did not converge\n{essentials.print_matrix(self.h_ks)}") from e
        self.y_a = generate_1rdm(self.Ns, self.Ne, self.wf_ks)
        if self.Ne > 0:
            e_homo = self.epsilon_s[(self.Ne - 1) // 2]
        else:
            e_homo = self.epsilon_s[0]
        e_lumo = self.epsilon_s[self.Ne // 2]
        if np.isclose(e_lumo, e_homo):
            raise errors.DegeneratedStatesError(self.Ne, self.epsilon_s)
        self.n_ks = np.copy(self.y_a.diagonal())

        if prevent_extreme_values:
            min_distance_from_extremes = 1e-3
            # In LPFET we don't want to have either fully occupied impurities or fully empty site because then non of
            # the chemical impurity potential will be able to match density from KS-DFT. for fully occupied orbital and
            # mu_imp == 50 we get occupation 1.99984983. 1e-3 is around 30% higher so this should solve the problem
            if np.any(np.isclose(self.n_ks, 0, atol=min_distance_from_extremes)):
                raise errors.EmptyFullOrbitalError(self.n_ks, self.n_ks[np.isclose(self.n_ks, 0, atol=1e-3)])

            if np.any(np.isclose(self.n_ks, 2, atol=min_distance_from_extremes)):
                raise errors.EmptyFullOrbitalError(self.n_ks, self.n_ks[np.isclose(self.n_ks, 2, atol=1e-3)])

    def casci(self, oscillation_compensation=0, v_hxc_0=None):
        mu_imp_first = np.nan
        first_iteration = True
        for site_group, group_tuple in enumerate(self.equiv_atom_groups):
            site_id = group_tuple[0]

            # Householder transforms impurity on index 0 so we have to make sure that impurity is on index 0:
            y_a_correct_imp = change_indices(self.y_a, site_id)
            t_correct_imp = change_indices(self.t, site_id)
            v_s_correct_imp = change_indices(self.v_s, site_id)

            p, v = qnb.tools.householder_transformation(y_a_correct_imp)
            h_tilde = p @ (t_correct_imp + np.diag(v_s_correct_imp)) @ p

            h_tilde_dimer = h_tilde[:2, :2]
            u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
            u_0_dimer[0, 0, 0, 0] += self.u[site_id]

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
            if v_hxc_0 is None:
                # This means that if we keep v_hxc_0 equal to None,
                # the algorithm does mu_imp -> v_hxc (mu_imp - mu_imp0 + mu_imp0)
                v_hxc_0 = mu_imp_first
            self.update_v_hxc(site_group, v_hxc_0 + delta_mu_imp, oscillation_compensation)
            self.embedded_mol.calculate_1rdm_spin_free()
            two_rdm = self.embedded_mol.calculate_2rdm_fh(index=0)
            on_site_repulsion_i = two_rdm[0, 0, 0, 0] * u_0_dimer[0, 0, 0, 0]
            v_term_repulsion_i = np.sum(two_rdm * v_tilde)
            for every_site_id in self.equiv_atom_groups[site_group]:
                self.imp_potential[every_site_id] = mu_imp
                self.v_term_repulsion[every_site_id] = v_term_repulsion_i
                self.kinetic_contributions[every_site_id] = 2 * h_tilde[1, 0] * self.embedded_mol.one_rdm[1, 0]
                self.onsite_repulsion[every_site_id] = on_site_repulsion_i

            first_iteration = False

    def update_v_hxc(self, site_group, mu_imp, oscillation_compensation):
        global COMPENSATION_5_FACTOR
        if type(oscillation_compensation) == int:
            oscillation_compensation = [oscillation_compensation]
        if len(self.v_hxc_progress) < 2:
            index = self.equiv_atom_groups[site_group][0]
            if len(self.v_hxc_progress) == 0:
                mu_minus_1 = self.v_hxc[index]
            else:
                mu_minus_1 = self.v_hxc_progress[-1][index]
            if 5 in oscillation_compensation:
                new_mu_imp = mu_minus_1
                new_mu_imp += np.tanh((mu_imp - mu_minus_1)) / COMPENSATION_5_FACTOR2
                general_logger.info(
                    f'(({site_group}): {mu_minus_1:.2f} {new_mu_imp - mu_minus_1:.2f} {mu_imp - mu_minus_1:.2f})',
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
                general_logger.info(
                    f'(({site_group}): {mu_minus_1:.2f} {new_mu_imp - mu_minus_1:.2f} {mu_imp - mu_minus_1:.2f},'
                    f' {self.compensation_ratio_dict[index]:.1f})',
                    end=', ')
                mu_imp = new_mu_imp
            if 1 in oscillation_compensation:
                if (mu_minus_2 - mu_minus_1) * (mu_minus_1 - mu_imp) < 0 and \
                        abs(mu_minus_2 - mu_minus_1) * 0.75 < abs(mu_minus_1 - mu_imp):
                    # First statement means that potential correction turned direction and second means that it is large
                    new_mu_imp = mu_minus_1 + (mu_imp - mu_minus_1) * COMPENSATION_1_RATIO
                    general_logger.info(f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f}', end=', ')
                    mu_imp = new_mu_imp
                    self.oscillation_correction_dict[(self.iteration_i, index)] = (
                        mu_minus_2, mu_minus_1, mu_imp, new_mu_imp)
            if 2 in oscillation_compensation:

                if (abs(pml[mu_counter] - pml[mu_same]) * 0.5 < abs(mu_minus_1 - mu_imp)) and \
                        ((mu_counter - mu_same) > 0):
                    # First statement means that potential correction turned direction and second means that it is large
                    new_mu_imp = mu_minus_1 + (mu_imp - mu_minus_1) * COMPENSATION_1_RATIO
                    general_logger.info(f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f}', end=', ')
                    mu_imp = new_mu_imp
                    self.oscillation_correction_dict[(self.iteration_i, index)] = (
                        mu_minus_2, mu_minus_1, mu_imp, new_mu_imp)
            if 3 in oscillation_compensation:
                pml2 = np.array(pml)
                x_data = np.range1(len(pml2)).reshape(-1, 1)
                reg = LinearRegression().fit(x_data, pml2)
                r2 = reg.score(x_data, pml2)
                factor1 = r2  # np.exp((r2 - 1) * 3)
                predicted = reg.predict([x_data[-1]])[0]
                new_mu_imp = factor1 * mu_imp + (1 - factor1) * predicted
                general_logger.info(
                    f'{mu_minus_2:.2f}->{mu_minus_1:.2f}->{new_mu_imp:.2f}!={mu_imp:.2f} ({r2}, {factor1})', end=', ')
                mu_imp = new_mu_imp

        for every_site_id in self.equiv_atom_groups[site_group]:
            self.v_hxc[every_site_id] = mu_imp

    def calculate_energy(self, silent=False):
        per_site_array = np.zeros(self.Ns, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),
                                                  ('v_term', float)])
        per_site_array['kin'] = self.kinetic_contributions
        per_site_array['v_ext'] = self.v_ext * self.n_ks
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
            general_logger.info(
                f'\n{"site":30s}{" ".join([f"{i:9d}" for i in range(self.Ns)])}{"total":>12s}\n{"Kinetic energy":30s}'
                f'{" ".join([f"{i:9.4f}" for i in self.kinetic_contributions])}{kinetic_contribution:12.7f}\n'
                f'{"External potential energy":30s}{" ".join([f"{i:9.4f}" for i in per_site_array["v_ext"]])}'
                f'{v_ext_contribution:12.7f}\n{"On-site repulsion":30s}'
                f'{" ".join([f"{i:9.4f}" for i in self.onsite_repulsion])}{u_contribution:12.7f}\n'
                f'{"V-term repulsion":30s}{" ".join([f"{i:9.4f}" for i in self.v_term_repulsion])}'
                f'{v_term_contribution:12.7f}\n'
                f'{"Occupations":30s}{" ".join([f"{i:9.4f}" for i in self.n_ks])}{np.sum(self.n_ks) * 2:12.7f}')
            general_logger.info(f'{"_" * 20}\nTotal energy:{total_energy}')

        return total_energy

    def compare_densities_fci(self, pass_object: Union[bool, class_qnb.HamiltonianV2] = False,
                              calculate_per_site=False) -> typing.Tuple[np.array, class_qnb.HamiltonianV2, tuple,
                                                                        np.array]:
        if type(pass_object) != bool:
            mol_full = pass_object
        else:
            mol_full = class_qnb.HamiltonianV2(self.Ns, self.Ne)
            mol_full.build_operator_a_dagger_a()
        u4d = np.zeros((self.Ns, self.Ns, self.Ns, self.Ns))
        for i in range(self.Ns):
            u4d[i, i, i, i] = self.u[i]
        v_tilde = None
        if self.v_term is not None:
            p = np.eye(self.Ns)
            v_tilde = np.einsum('ip, iq, jr, js, ij -> pqrs', p, p, p, p, self.v_term)
        mol_full.build_hamiltonian_fermi_hubbard(self.t + np.diag(self.v_ext), u4d, v_term=v_tilde)
        mol_full.diagonalize_hamiltonian()
        y_ab = mol_full.calculate_1rdm_spin_free()
        densities = y_ab.diagonal()
        kinetic_contribution = np.sum(y_ab * self.t)
        v_ext_contribution = np.sum(self.v_ext * densities)
        total_energy = mol_full.eig_values[0]
        u_contribution = total_energy - kinetic_contribution - v_ext_contribution

        if calculate_per_site:
            per_site_array = np.zeros(self.Ns, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),
                                                      ('v_term', float)])

            two_rdm_u = mol_full.build_2rdm_fh_on_site_repulsion(u4d)
            if self.v_term is not None:
                two_rdm_v_term = mol_full.build_2rdm_fh_dipolar_interactions(v_tilde)
                on_site_repulsion_array = two_rdm_v_term * v_tilde
                for site in range(self.Ns):
                    per_site_array[site]['v_term'] = 0.5 * (np.sum(on_site_repulsion_array[site, site, :, :]) +
                                                            np.sum(on_site_repulsion_array[:, :, site, site]))
            t_multiplied_matrix = y_ab * self.t
            for site in range(self.Ns):
                per_site_array[site]['v_ext'] = self.v_ext[site] * densities[site]
                per_site_array[site]['u'] = two_rdm_u[site, site, site, site] * self.u[site]
                per_site_array[site]['kin'] = np.sum(t_multiplied_matrix[site])
                per_site_array[site]['tot'] = sum(per_site_array[site])
            u_contribution = np.sum(per_site_array['u'])
            v_term_contribution = np.sum(per_site_array['v_term'])

            return y_ab, mol_full, (total_energy, kinetic_contribution, v_ext_contribution,
                                    u_contribution, v_term_contribution), per_site_array
        general_logger.info("FCI densities (per spin):", densities)
        general_logger.info(f'FCI energy: {mol_full.eig_values[0]}')
        return y_ab, mol_full, (total_energy, kinetic_contribution, v_ext_contribution,
                                u_contribution, 0), np.array([])

    def plot_density_evolution(self):
        self.density_progress = np.array(self.density_progress)
        for i in range(self.density_progress.shape[1]):
            plt.plot(self.density_progress[:, i])
        plt.xlabel("Iteration")
        plt.ylabel("Density")
        plt.title("Evolution of density in simulation")
        plt.show()

    def plot_hubbard_molecule(self):
        g = nx.Graph()
        colors = ['lightgrey', 'mistyrose', 'lightcyan', 'thistle', 'springgreen', 'yellow', 'cyan', 'magenta',
                  'orange']
        color_map = []
        label_dict = {}
        edge_labels = dict()
        for i in range(self.Ns):
            node_string = f"U={self.u[i]:.1f}\nv_ext={self.v_ext[i]:.3f}"
            for key, value in enumerate(self.equiv_atom_groups):
                if i in value:
                    group_id = key
                    break
            else:
                group_id = -1
            g.add_nodes_from([(i, {'color': colors[group_id]})])
            label_dict[i] = node_string
            for j in range(i, self.Ns):
                if self.t[i, j] != 0:
                    edge_string = f"{self.t[i, j]}"
                    g.add_edge(i, j)
                    edge_labels[(i, j)] = edge_string
        for i in g.nodes:
            color_map.append(g.nodes[i]['color'])
        fig, ax = plt.subplots(1, 1)
        position = nx.spring_layout(g)
        nx.draw(g, pos=position, ax=ax, labels=label_dict, with_labels=True, node_color=color_map, node_size=5000,
                font_weight='bold')
        nx.draw_networkx_edge_labels(g, position, edge_labels)
        ax.set_xlim(*np.array(ax.get_xlim()) * 1.3)
        ax.set_ylim(*np.array(ax.get_ylim()) * 1.3)
        general_logger.info(f"results/{self.description}_molecule.png")
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
        self.equiv_atom_groups = []
        self.v_hxc = np.zeros(self.Ns, dtype=np.float64)  # Hartree exchange correlation potential
        self.density_progress = []  # This object is used for gathering changes in the density over iterations
        self.v_hxc_progress = []

        self.oscillation_correction_dict = dict()

        self.optimize_progress_input = []
        self.optimize_progress_output = []

    def update_variables_embedded(self, v_tilde, h_tilde, site_group, mu_imp, embedded_mol, u_0_dimer=None):
        two_rdm_u = embedded_mol.build_2rdm_fh_on_site_repulsion(u_0_dimer)
        on_site_repulsion_i = two_rdm_u[0, 0, 0, 0] * u_0_dimer[0, 0, 0, 0]
        if self.v_term is not None:
            two_rdm_v_term = embedded_mol.build_2rdm_fh_dipolar_interactions(v_tilde)
            v_term_repulsion_i = np.sum(two_rdm_v_term * v_tilde)
        else:
            v_term_repulsion_i = 0
        for every_site_id in self.equiv_atom_groups[site_group]:
            self.kinetic_contributions[every_site_id] = h_tilde[1, 0] * (embedded_mol.one_rdm[1, 0])
            self.onsite_repulsion[every_site_id] = on_site_repulsion_i
            self.imp_potential[every_site_id] = mu_imp
            self.v_term_repulsion[every_site_id] = v_term_repulsion_i
        # something like:
        # self.imp_potential[self.blocks[site_group]] = mu_imp
        # But now we see that there is a problem with the sites and blocks :c

    def update_variables_embedded_block(self, v_tilde, h_tilde, site_group, mu_imp,
                                        embedded_mol: class_qnb.HamiltonianV2, u_tilde):
        two_rdm_on_site = embedded_mol.build_2rdm_fh_on_site_repulsion(u_tilde)
        for one_site_id, one_site in enumerate(site_group):
            equivalent_atoms = [i for i in mol_obj.equiv_atoms_in_block if one_site in i][0]
            v_term_one = self.transform_v_term(p, site_group, one_site)
            v_term_repulsion_i = np.sum(embedded_mol.build_2rdm_fh_dipolar_interactions(v_term_one) * v_term_one)
            on_site_repulsion_i = two_rdm_on_site[one_site_id, one_site_id, one_site_id, one_site_id] * \
                                  u_tilde[one_site_id, one_site_id, one_site_id, one_site_id]
            for every_site_id in equivalent_atoms:
                # Kinetic contribution???
                self.kinetic_contributions[every_site_id] = 2 * h_tilde[1, 0] * embedded_mol.one_rdm[1, 0]
                self.onsite_repulsion[every_site_id] = on_site_repulsion_i
                self.imp_potential[every_site_id] = mu_imp
                self.v_term_repulsion[every_site_id] = v_term_repulsion_i

    def transform_v_term(self, p, site_id, calculate_one_site=-1):
        if isinstance(site_id, (int, np.int32, np.int64)):
            site_id = [site_id]
        impurity_size = len(site_id)
        if self.v_term is not None:
            v_term = np.zeros((self.Ns, self.Ns))
            if calculate_one_site == -1:
                change_id_obj = site_id
            else:
                change_id_obj = calculate_one_site
            v_term[change_id_obj, :] += self.v_term[change_id_obj, :] / 2
            v_term[:, change_id_obj] += self.v_term[:, change_id_obj] / 2
            if site_id == list(range(impurity_size)):
                v_term_correct_indices = v_term
            else:
                mask = np.arange(self.Ns)
                mask[range(impurity_size)] = site_id
                mask[site_id] = range(impurity_size)
                v_term_correct_indices = v_term[:, mask][mask, :]
            v_tilde = np.einsum('ip, iq, jr, js, ij -> pqrs', p, p, p, p,
                                v_term_correct_indices)[:2 * impurity_size, :2 * impurity_size, :2 * impurity_size,
                      :2 * impurity_size]
        else:
            v_tilde = None
        return v_tilde


def cost_function_whole(v_hxc_approximation: np.array, mol_obj: Molecule) -> np.array:
    mol_obj.v_hxc = np.zeros(mol_obj.Ns, float)

    if mol_obj.block_hh:
        eq_atom = mol_obj.equiv_atoms_in_block
    else:
        eq_atom = mol_obj.equiv_atom_groups

    for group_id, group_site_tuple in enumerate(eq_atom):
        if group_id != 0:
            for site_id_i in group_site_tuple:
                mol_obj.v_hxc[site_id_i] = v_hxc_approximation[group_id - 1]
    mol_obj.v_hxc_progress.append(mol_obj.v_hxc)
    mol_obj.calculate_ks(True)
    mol_obj.density_progress.append(mol_obj.n_ks)
    output_array = np.zeros(len(eq_atom) - 1, float)
    mu_imp_first = np.nan
    first_iteration = True
    output_index = 0

    for site_group, group_tuple in enumerate(mol_obj.equiv_atom_groups):
        site_id = group_tuple[0]
        y_a_correct_imp = change_indices(mol_obj.y_a, site_id)
        p, v = qnb.tools.householder_transformation(y_a_correct_imp)
        if np.any(np.isnan(p)):
            raise errors.HouseholderTransformationError(y_a_correct_imp)
        h_tilde = p @ (change_indices(mol_obj.t, site_id) + np.diag(change_indices(mol_obj.v_s, site_id))) @ p
        h_tilde_dimer = h_tilde[:2, :2]
        v_tilde = mol_obj.transform_v_term(p, site_id)
        u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
        u_0_dimer[0, 0, 0, 0] += mol_obj.u[site_id]
        mol_obj.h_tilde_dimer[site_group] = h_tilde_dimer
        if first_iteration:
            if 0 not in mol_obj.equiv_atom_groups[site_group]:
                raise Exception("Unexpected behaviour: First impurity site should have been the 0th site")
            sol = scipy.optimize.root_scalar(cost_function_casci_root,
                                             args=(
                                                 mol_obj.embedded_mol, h_tilde_dimer, u_0_dimer, mol_obj.n_ks[site_id],
                                                 v_tilde),
                                             bracket=[-50, 50], method='brentq', options={'xtol': 1e-6})
            mu_imp, function_calls = sol.root, sol.function_calls
            mu_imp_first = mu_imp
        else:
            mu_imp = mu_imp_first + mol_obj.v_hxc[site_id]
            error_i = cost_function_casci_root(mu_imp, mol_obj.embedded_mol, h_tilde_dimer, u_0_dimer,
                                               mol_obj.n_ks[site_id], v_tilde)
            output_array[output_index] = error_i
            output_index += 1
        mol_obj.update_variables_embedded(v_tilde, h_tilde, site_group, mu_imp, mol_obj.embedded_mol, u_0_dimer)
        first_iteration = False
    rms = np.sqrt(np.mean(np.square(output_array)))
    general_logger.info(
        f"for input {''.join(['{num:{dec}}'.format(num=cell, dec='+10.2e') for cell in mol_obj.v_hxc])} error is"
        f" {''.join(['{num:{dec}}'.format(num=cell, dec='+10.2e') for cell in output_array])} "
        f" (RMS = {rms})")
    return output_array


def cost_function_whole_block(v_hxc_approximation: np.array, mol_obj: Molecule) -> np.array:
    global ROOT_FINDING_LIST_INPUT, ROOT_FINDING_LIST_OUTPUT
    temp1 = ''.join(['{num:{dec}}'.format(num=cell, dec='+10.3f') for cell in mol_obj.v_hxc])
    general_logger.log(15, f"| Start of cost function 1: Input = {temp1}")
    mol_obj.optimize_progress_input.append(v_hxc_approximation.copy())
    mol_obj.v_hxc = np.zeros(mol_obj.Ns, float)
    for group_id, group_site_tuple in enumerate(mol_obj.equiv_atom_groups):
        if group_id != 0:
            for site_id_i in group_site_tuple:
                mol_obj.v_hxc[site_id_i] = v_hxc_approximation[group_id - 1]
    mol_obj.v_hxc_progress.append(mol_obj.v_hxc)
    mol_obj.calculate_ks()
    mol_obj.density_progress.append(mol_obj.n_ks)
    output_array_non_reduced = np.zeros(mol_obj.Ns, float) * np.nan
    output_array = np.zeros(len(mol_obj.equiv_atom_groups) - 1, float) * np.nan
    mu_imp_first = np.nan
    first_iteration = True

    for site_group, group_tuple in enumerate(mol_obj.equiv_block_groups):
        block_id = group_tuple[0]
        site_id = mol_obj.blocks[block_id]
        block_size = len(site_id)
        general_logger.log(10, f"|| Site id: {site_id}")
        y_a_correct_imp = change_indices(mol_obj.y_a, site_id)
        p, moore_penrose_inv = qnb.tools.block_householder_transformation(y_a_correct_imp, block_size)
        if np.any(np.isnan(p)):
            raise errors.HouseholderTransformationError(y_a_correct_imp)
        t_correct_indices = change_indices(mol_obj.t, site_id)
        h_tilde = p @ (t_correct_indices + np.diag(change_indices(mol_obj.v_s, site_id))) @ p
        h_tilde_dimer = h_tilde[:block_size * 2, :block_size * 2]
        v_tilde = mol_obj.transform_v_term(p, site_id)
        u_0_dimer = np.zeros((block_size * 2, block_size * 2, block_size * 2, block_size * 2), dtype=np.float64)
        range1 = np.arange(len(site_id))
        u_0_dimer[range1, range1, range1, range1] += mol_obj.u[site_id]
        if first_iteration:
            general_logger.log(10, f"||| First cluster: We want to get densities {mol_obj.n_ks[site_id]}")
            if 0 not in site_id:
                raise Exception("Unexpected behaviour: First impurity site should have been the 0th site")
            try:
                ROOT_FINDING_LIST_INPUT = []
                ROOT_FINDING_LIST_OUTPUT = []
                model = scipy.optimize.root(cost_function_casci_root, mol_obj.v_hxc[site_id],
                                            args=(mol_obj.embedded_mol_dict[block_size], h_tilde_dimer, u_0_dimer,
                                                  mol_obj.n_ks[site_id], v_tilde),
                                            options={'fatol': 1e-4, 'maxfev': 40, 'fnorm': abs_norm, 'ftol': 0},
                                            method='df-sane')
            except FloatingPointError as e:
                general_logger.info(e)
                model = scipy.optimize.OptimizeResult()
                model.success = False
                model.x = mol_obj.v_hxc[site_id]

            if not model.success or np.sum(np.square(model.fun)) > 1e-3:
                general_logger.info("||| Embedded cluster: Didn't manage to converge with df-sane;"
                                    " trying with hybr root finder")
                # Second chance with another model
                model = scipy.optimize.root(cost_function_casci_root, mol_obj.v_hxc[site_id],
                                            args=(mol_obj.embedded_mol_dict[block_size], h_tilde_dimer, u_0_dimer,
                                                  mol_obj.n_ks[site_id], v_tilde), options={'eps': 0.01}, method='hybr')
            if not model.success or np.sum(np.square(model.fun)) > 1e-3:
                general_logger.info("||| Didn't converge :c ")
                root_finding_list_output = np.array(ROOT_FINDING_LIST_OUTPUT)
                root_finding_list_input = np.array(ROOT_FINDING_LIST_INPUT)
                weights = np.sqrt(np.sum(np.square(1 / (root_finding_list_output + 1e-5)), axis=1))
                mask = weights > np.percentile(weights, 95)
                final_approximation1 = np.average(root_finding_list_input[mask], axis=0, weights=weights[mask])
                error1 = cost_function_casci_root(final_approximation1, mol_obj.embedded_mol_dict[block_size],
                                                  h_tilde_dimer, u_0_dimer,
                                                  mol_obj.n_ks[site_id], v_tilde)
                final_approximation2 = root_finding_list_input[weights.argsort()[::-1]][0]
                error2 = cost_function_casci_root(final_approximation2, mol_obj.embedded_mol_dict[block_size],
                                                  h_tilde_dimer, u_0_dimer,
                                                  mol_obj.n_ks[site_id], v_tilde)
                if abs_norm(error2) > abs_norm(error1):
                    final_approximation = final_approximation1
                    final_error = error1
                else:
                    final_approximation = final_approximation2
                    final_error = error2
                # plt.quiver(ROOT_FINDING_LIST_INPUT[:, 0], ROOT_FINDING_LIST_INPUT[:, 1], ROOT_FINDING_LIST_OUTPUT[:, 0],
                #            ROOT_FINDING_LIST_OUTPUT[:, 1], scale=1)
                # plt.xlim(3.35, 3.43)
                # plt.ylim(2.125, 2.175)
                # plt.show()
                if abs_norm(final_error) < 1e-2:
                    model = scipy.optimize.OptimizeResult()
                    model.success = True
                    model.x = final_approximation
                else:
                    raise Exception(f'Inversion of a cluster did not succeed'
                                    f'\n{essentials.print_matrix(h_tilde_dimer, ret=True)}\n'
                                    f'Desired density: {mol_obj.n_ks[site_id]}\n best guess error was: {final_error}')
            zero_index_in_block = list(site_id).index(0)
            mu_imp = model.x
            mu_imp_first = mu_imp[zero_index_in_block]

        mu_imp = mu_imp_first + mol_obj.v_hxc[site_id]
        error_i = cost_function_casci_root(mu_imp, mol_obj.embedded_mol_dict[block_size], h_tilde_dimer, u_0_dimer,
                                           mol_obj.n_ks[site_id], v_tilde)
        general_logger.log(10, f"||| Higher accuary method gave error = {error_i} for mu_imp={mu_imp}")
        output_array_non_reduced[site_id] = error_i

        for index1, one_site_id in enumerate(site_id):
            eq_block = find_equivalent_block(mol_obj, one_site_id)
            output_array_non_reduced[eq_block] = error_i[index1]

        # energy contributions
        one_rdm_c = mol_obj.embedded_mol_dict[block_size].one_rdm
        two_rdm_c = mol_obj.embedded_mol_dict[block_size].build_2rdm_fh_on_site_repulsion(u_0_dimer)
        for index, site in enumerate(site_id):
            t_tilde_i = t_correct_indices.copy()
            t_tilde_i[[i for i in range(mol_obj.Ns) if i != index]] = 0
            t_tilde_i = (p @ t_tilde_i @ p)[:block_size * 2, :block_size * 2]

            # essentials.print_matrix((t_tilde_i * one_rdm_c))
            eq_block = find_equivalent_block(mol_obj, site)
            mol_obj.kinetic_contributions[eq_block] = np.sum(t_tilde_i * one_rdm_c)
            mol_obj.onsite_repulsion[eq_block] = two_rdm_c[index, index, index, index] * u_0_dimer[
                index, index, index, index]

        # mol_obj.update_variables_embedded_block(v_tilde, h_tilde, site_group, mu_imp[one_site_id],
        #                                         mol_obj.embedded_mol_dict[block_size], u_0_dimer)
        first_iteration = False
    if np.any(np.isnan(output_array_non_reduced)):
        raise Exception("Didn't cover all groups")
    for group_id, group_site_tuple in enumerate(mol_obj.equiv_atom_groups):
        if group_id != 0:
            # now in output_array_non_reduced we have many nan_values because we don't want to calculate for every
            # site but just for one in equivalent sites. Sometimes it still happens that we have more than
            # 1 calculation. In this case we want to have same value
            values_from_same_group = output_array_non_reduced[list(group_site_tuple)]
            values_from_same_group = values_from_same_group[np.logical_not(np.isnan(values_from_same_group))]
            output_array[group_id - 1] = np.mean(values_from_same_group)
    max_dev = np.max(np.abs(output_array))
    general_logger.log(20,
                       f"| End of cost function 1: for input {''.join(['{num:{dec}}'.format(num=cell, dec='+10.3f') for cell in mol_obj.v_hxc])}"
                       f" error is {''.join(['{num:{dec}}'.format(num=cell, dec='+10.2e') for cell in output_array])} "
                       f" (max deviation = {max_dev})")
    mol_obj.optimize_progress_output.append(output_array.copy())
    return output_array


def find_equivalent_block(mol_obj: Molecule, one_site_id: int):
    eq_block = None
    for t1 in range(len(mol_obj.equiv_atoms_in_block)):
        if one_site_id in mol_obj.equiv_atoms_in_block[t1]:
            eq_block = mol_obj.equiv_atoms_in_block[t1]
            break
    if eq_block is None:
        raise Exception(f"Site not found in the equiv_atoms_in_block ({one_site_id} in {mol_obj.equiv_atoms_in_block})")
    return eq_block


def cost_function_casci_root(mu_imp, embedded_mol, h_tilde_dimer, u_0_dimer, desired_density, v_tilde):
    # mu_imp = mu_imp[0]
    global ITERATION_NUM, ROOT_FINDING_LIST_INPUT, ROOT_FINDING_LIST_OUTPUT
    ROOT_FINDING_LIST_INPUT.append(mu_imp.copy())
    cluster_size = h_tilde_dimer.shape[0]
    half_diagonal = np.arange(cluster_size / 2, dtype=int)
    ITERATION_NUM += 1
    mu_imp_array = np.zeros((cluster_size, cluster_size))
    mu_imp_array[half_diagonal, half_diagonal] = mu_imp
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer - mu_imp_array, u_0_dimer, v_term=v_tilde)
    embedded_mol.diagonalize_hamiltonian()
    density_dimer = embedded_mol.calculate_1rdm_spin_free(index=0)
    result = density_dimer[half_diagonal, half_diagonal] - desired_density
    ROOT_FINDING_LIST_OUTPUT.append(result.copy())
    general_logger.log(5, f"|||| cost function 2: Input: {mu_imp}; Output: {result}; desired density: {desired_density}")
    return result


def see_landscape_ruggedness(embedded_mol, h_tilde_dimer, u_0_dimer, goal_density=False, optimized_potential=False,
                             num_dim=1, range1=(-2, 2 + 0.1, 0.1)):
    if num_dim == 1:
        x = np.arange(*range1)
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
    for id1, params in sites.items():
        if 'U' in params:
            u[id1] = params['U']
        elif 'u' in params:
            u[id1] = params['u']
        else:
            raise "Problem with params: " + params
        v[id1] = params['v']
    for pair, param in connections.items():
        t[pair[0], pair[1]] = -param
        t[pair[1], pair[0]] = -param
    return t, v, u


# CALCULATE JACOBIAN MATRIX
#
# plus_list = []
# minus_list = []
# for i in range(7):
#     add = np.zeros(7)
#     add[i] += 0.1
#     original = cost_function_whole_block(starting_approximation, self)
#     plus = cost_function_whole_block(starting_approximation + add, self)
#     minus = cost_function_whole_block(starting_approximation - add, self)
#     plus_list.append((plus - original) / 0.1)
#     minus_list.append((original - minus) / 0.1)
# plus_list = np.array(plus_list).T
# minus_list = np.array(minus_list).T
# essentials.print_matrix(plus_list)
# essentials.print_matrix(minus_list)

if __name__ == "__main__":
    pass
