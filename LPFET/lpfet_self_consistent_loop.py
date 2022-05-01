from LPFET.lpfet import MoleculeBare

COMPENSATION_1_RATIO = 0.5  # for the Molecule.update_v_hxc
COMPENSATION_MAX_ITER_HISTORY = 4
COMPENSATION_5_FACTOR = 1
COMPENSATION_5_FACTOR2 = 0.5


class MoleculeSCL(MoleculeBare):
    def __init__(self):
        super().__init__(site_number, electron_number, description)
        self.embedded_mol_dict = dict()
        self.compensation_ratio_dict = dict()

    def add_parameters(self, u, t, v_ext,
                       v_term_repulsion_ratio: typing.Union[bool, float] = False):
        super(MoleculeSCL, self).add_parameters(u, t, v_ext, v_term_repulsion_ratio)
        for index, item in enumerate(equiv_atom_group_list):
            self.compensation_ratio_dict[index] = COMPENSATION_5_FACTOR

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

            sol = sc_opt.root_scalar(mol_obj.cost_function_2,
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

    def clear_object(self, description=''):
        super().clear_object(description)
        self.oscillation_correction_dict = dict()