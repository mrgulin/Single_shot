import numpy as np
import matplotlib.pyplot as plt
OUTPUT_FORMATTING_NUMBER = "+12.6f"
OUTPUT_SEPARATOR = "  "


def print_matrix(matrix, plot_heatmap='', ret=False):
    ret_string = ""
    for line in matrix:
        l1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in line]
        ret_string += f'{OUTPUT_SEPARATOR}'.join(l1) + "\n"
    if plot_heatmap:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.title(plot_heatmap)
    if ret:
        return ret_string
    print(ret_string, end='')


def calculate_hopping(t_tilde, energy, big_u, delta_v):
    up = (-1.0 / 4.0) * (8.0 * t_tilde * (energy - big_u))
    return up / (- 4.0 * (t_tilde ** 2) + big_u ** 2 - delta_v ** 2 - 4.0 * big_u * energy + 3.0 * (energy ** 2))


def calculate_delta_v_pr(t_tilde, energy, big_u, delta_v):
    denominator = (- 4.0 * (t_tilde ** 2) + big_u ** 2.0 - delta_v ** 2 - 4.0 * big_u * energy + 3.0 * (energy ** 2.0))
    return (2.0 * delta_v * energy) / denominator


def calculate_d_occ(t_tilde, energy, big_u, delta_v):
    denominator = (- 4.0 * (t_tilde ** 2) + big_u ** 2 - delta_v ** 2 - 4.0 * big_u * energy + 3.0 * (energy ** 2))
    return (- 2.0 * (t_tilde ** 2.0) - big_u * energy + energy ** 2 - delta_v * energy) / denominator
    # return (- 2.0 * t_tilde * t_tilde) - big_u * energy + energy ** 2 - 4.0 * big_u * energy + 3.0 * energy * energy


def calculate_energy(U, U1, t_tilde, delta_v):
    big_u = 0.5 * (U1 + U)
    u = big_u / (2 * t_tilde)
    nu = delta_v / (2 * t_tilde)
    w = np.sqrt(3 * (1 + nu * nu) + u * u)
    theta = (1.0 / 3.0) * np.arccos((9 * (nu * nu - 0.5) - u * u) * (u / (w * w * w)))
    energy = (4.0 / 3.0 * t_tilde) * (u - w * np.sin(theta + (np.pi / 6.0)))
    return big_u, energy


class Householder:
    def __init__(self, particle_number: int, electron_number: int, u: float, debug=False, skip_unnecessary=False):
        self.N = particle_number
        self.Ne = electron_number

        self.t = 1.00
        self.U = u
        self.gamma = np.zeros((self.N, self.N), dtype=np.float64)
        self.gamma_tilde = np.zeros((self.N, self.N), dtype=np.float64)
        self.h = np.zeros((self.N, self.N), dtype=np.float64)  # non-interacting hamiltonian h_ij

        self.P = np.zeros((self.N, self.N), dtype=np.float64)  # Householder transformation matrix
        self.v = np.zeros(self.N, dtype=np.float64)  # Householder vector v

        self.procedure_log = ""
        self.results_string = ''
        self.combined_results_string = {'col_names': '', 'row': ''}

        self.vars = {'hopping': [0.0, 0.0], 'density': [0.0, 0.0], 'd_occ': [0.0, 0.0], 'delta_v': 0.0, 'epsilon': 0.0,
                     'mu_imp': 0.0, 't_tilde': 0.0, 'N_electron_cluster': 0.0}

        self.mu = {'KS': 0.0, 'imp': 0.0, 'ext': 0.0}
        self.e_site = {"main": 0.0, 'without_mu_opt': 0.0, 'type3': 0.0, 'type4': 0.0}

        self.skip_unnecessary = skip_unnecessary
        self.debug = debug

        self.ei_val = None
        self.ei_vec = None

        self.lieb_min_list = []

    def calculate_one(self):
        """
        Equivalent to DENSITY_MATRIX subroutine
        """
        if (self.Ne % 2) != 0:
            raise 'Problem! Number of electrons is not even!'

        n = float(self.Ne) / float(self.N)  # Density

        print(f'calculation with Ns={self.N}, Ne={self.Ne}, density={n}')

        # Huckel hamiltonian generation: self.h in our case
        self.h = self.generate_huckel_hamiltonian()

        # Generating eigenvalues and eigenvectors
        ei_val, ei_vec = np.linalg.eig(self.h)  # v[:,i] corresponds to eigval w[i]
        idx = ei_val.argsort()[::1] # Sorting matrix and vector by ascending order
        ei_val = ei_val[idx]
        ei_vec = ei_vec[:, idx]
        if not self.skip_unnecessary:
            self.ei_vec = ei_vec
            self.ei_val = ei_val

        # generation of 1RDM
        self.gamma = np.zeros((self.N, self.N), dtype=np.float64)  # reset gamma
        # for Ne_cnt in range(0, Ne, 2): then we would have k goes from 0 to ...
        for k in range(int(self.Ne / 2)):  # go through all orbitals that are occupied!
            for i in range(self.N):
                for j in range(self.N):
                    self.gamma[i, j] += ei_vec[i, k] * ei_vec[j, k]
        mu_ks = ei_val[(self.Ne - 1) // 2]

       # TODO: maybe generate pictures of matrices
        # self.procedure_log += f" = {mu_KS}\n\nGAMMA0 \n{print_matrix(self.gamma, False, True)}\n"
        self.mu['KS'] = mu_ks

        # Householder vector generation
        self.generate_householder_vector()

        # Calculate variables
        self.calculate_variables()

        # do a minimization
        self.lieb_maximization()

        self.mu['ext'] = self.mu['KS'] + self.mu["imp"]
        self.e_site["without_mu_opt"] = 4.0 * self.t * (1.0 - 2.0 * (self.v[1] ** 2)) * self.vars['hopping'][0] + \
                                        self.U * self.vars['d_occ'][0]
        self.e_site["main"] = 4.0 * self.t * (1.0 - 2.0 * (self.v[1] ** 2)) * self.vars['hopping'][1] + self.U * \
                              self.vars['d_occ'][1]
        self.e_site["type3"] = - 4.0 * self.t * self.gamma[0, 1] + self.U * self.vars['d_occ'][0]
        # Type3 is exact in the case of non-interacting electrons
        self.e_site["type4"] = self.e_site["main"] + self.U * (1.0 - n)
        self.write_report(False)

        if self.debug:
            self.procedure_log += f"CALCULATIONS MADE FOR THE KS SYSTEM WITH Ns = {self.N} and Ne = {self.Ne} ==>"
            self.procedure_log += f"DENSITY = {n}\n\n"
            self.procedure_log += "Huckel_hamiltonian\n" + print_matrix(self.h, False, True) + '\n\n'
            self.procedure_log += "Eigenvectors\n" + print_matrix(ei_vec, False, True) + '\n\n'
            self.procedure_log += "Eigenvalues\n" + "".join([f"{i:{OUTPUT_FORMATTING_NUMBER}}" for i in ei_val]) + '\n'
            self.procedure_log += "Gamma_0\n" + print_matrix(self.gamma, False, True) + '\n\n'
            self.procedure_log += "HH_vec\n" + "".join([f"{i:{OUTPUT_FORMATTING_NUMBER}}" for i in self.v]) + '\n'
            data_file = open(f"Procedure_log-U-{self.U:.0f}_N-{self.N}.txt", 'w', encoding='UTF-8')
            data_file.write(self.procedure_log)
            data_file.close()

    def generate_huckel_hamiltonian(self):
        h = np.zeros((self.N, self.N), dtype=np.float64)  # reinitialization
        t = self.t
        if (self.Ne / 2) % 2 == 0:
            h[0, self.N - 1] = t
            h[self.N - 1, 0] = t
            self.procedure_log += "ANTIPERIODIC" + '\n'
        else:
            h[0, self.N - 1] = -t
            h[self.N - 1, 0] = -t
            self.procedure_log += "PERIODIC" + '\n'

        h += np.diag(np.full((self.N - 1), -t), -1) + np.diag(np.full((self.N - 1), -t), 1)
        return h

    def generate_householder_vector(self):
        sum_m = 0
        # SHIFTED INDICES!!
        for j in range(1, self.N):
            sum_m += self.gamma[j, 0] * self.gamma[j, 0]
        alpha = -1 * np.sign(self.gamma[1, 0]) * np.sqrt(sum_m)  # in notes it is xi
        r = np.sqrt(0.5 * alpha * (alpha - self.gamma[1, 0]))

        self.v = np.zeros((self.N,), dtype=np.float64)  # reset array, v[0] = 0 so it is okay
        self.gamma_tilde = np.zeros((self.N, self.N), dtype=np.float64)
        self.P = np.zeros((self.N, self.N), dtype=np.float64)  # Householder transformation matrix

        self.v[1] = (self.gamma[1, 0] - alpha) / (2. * r)
        for i in range(2, self.N):
            if self.gamma[i, 0] == 0:
                self.v[i] = 0
            else:
                self.v[i] = self.gamma[i, 0] / (2 * r)

        if not self.skip_unnecessary:
            for i in range(self.N):
                for j in range(self.N):
                    self.P[i, j] = int(i == j) - 2.0 * self.v[i] * self.v[j]

            self.gamma_tilde = self.P @ self.gamma @ self.P

        l1 = ['{num:{dec}}'.format(num=i, dec=OUTPUT_FORMATTING_NUMBER) for i in self.v]
        self.procedure_log += f"{f'{OUTPUT_SEPARATOR}'.join(l1)}\n\n"

    def calculate_variables(self):
        self.vars = {'hopping': [0, 0], 'density': [0, 0], 'd_occ': [0, 0], 'delta_v': 0, 'epsilon': 0, 'mu_imp': 0,
                     't_tilde': 0, 'N_electron_cluster': 0}
        N = self.N  # just for shorter code
        t_tilde = self.t + 2 * self.v[1] * (self.v[1] * self.h[0, 1] + self.v[N - 1] * self.h[0, N - 1])
        t_tilde = - t_tilde
        U1 = 0
        epsilon_vector = [0, 0, 0]
        for i in range(self.N):
            for j in range(self.N):
                epsilon_vector[0] += self.h[i, j] * self.v[i] * self.v[j]
            epsilon_vector[1] += self.v[i] * self.h[1, i]
        epsilon_vector[2] = 4 * self.v[1] * (self.v[1] * epsilon_vector[0] - epsilon_vector[1])
        epsilon = epsilon_vector[2]

        delta_v = epsilon + 0.5 * (U1 - self.U)

        big_u, energy = calculate_energy(self.U, U1, t_tilde, delta_v)

        self.vars['d_occ'][0] = calculate_d_occ(t_tilde, energy, big_u, delta_v)
        self.vars['hopping'][0] = calculate_hopping(t_tilde, energy, big_u, delta_v)
        delta_v_pr = calculate_delta_v_pr(t_tilde, energy, big_u, delta_v)
        self.vars['density'][0] = 1.0 - delta_v_pr

        self.vars['delta_v'] = delta_v
        self.vars['epsilon'] = epsilon
        self.vars['epsil_v'] = epsilon_vector
        self.vars['U1'] = U1
        self.vars['t_tilde'] = t_tilde
        self.vars['N_electron_cluster'] = self.vars['density'][0] * self.N

    def lieb_maximization(self, ):
        n = [0, 0]
        n[0] = self.Ne / self.N
        delta_v = None
        F_lieb = -1000
        self.lieb_min_list = []
        # TODO: Is there no better way to do this maximization (steepest descend?)
        for delta_v_lieb in np.arange(-15, 15, 1/200):
            if self.vars['t_tilde'] == 0:
                print("Looks like you are trying to do minimization before variable calculation!")
            big_u, energy = calculate_energy(self.U, self.vars['U1'], self.vars['t_tilde'], delta_v_lieb)
            F_lieb_current = energy + delta_v_lieb * (n[0] - 1.0)  # * 2 / 2 and also
            self.lieb_min_list.append([delta_v_lieb, energy, F_lieb_current])
            if F_lieb_current > F_lieb:
                F_lieb = F_lieb_current
                delta_v = delta_v_lieb
                n[1] = n[0]
        self.procedure_log += f""

        big_u, energy = calculate_energy(self.U, self.vars['U1'], self.vars['t_tilde'], delta_v)

        self.vars['d_occ'][1] = calculate_d_occ(self.vars['t_tilde'], energy, big_u, delta_v)
        self.vars['hopping'][1] = calculate_hopping(self.vars['t_tilde'], energy, big_u, delta_v)
        delta_v_pr = calculate_delta_v_pr(self.vars['t_tilde'], energy, big_u, delta_v)
        self.vars['density'][1] = 1.0 - delta_v_pr
        self.vars["KE"] = 2.0 * self.vars['t_tilde'] * self.vars['hopping'][1]

        mu_imp = - self.vars['epsilon'] - 0.5 * (self.vars['U1'] - self.U) + delta_v
        self.vars['mu_imp'] = mu_imp
        self.mu['imp'] = mu_imp
        self.vars['delta_v'] = delta_v

    def write_report(self, print_result=True):
        # Result_string part
        n = float(self.Ne) / float(self.N)  # Density
        col1 = ['Ns', 'Ne', 'Density', 'μ_KS', 'μ_imp', 'μ_ext', 'Impurity occupation',
                'gamma_01 from Hamiltonian', 'KE', 'D_occ from Hamiltonian', 't_tilde', 'epsilon']
        col2 = [self.N, self.Ne, n, self.mu['KS'], self.mu['imp'], self.mu['ext'], self.vars['density'],
                self.vars['hopping'], self.vars["KE"], self.vars['d_occ'], self.vars["t_tilde"],
                self.vars["epsilon"]]
        max_col1 = 20
        for i in range(len(col1)):
            if type(col2[i]) != list:
                self.results_string += f"{col1[i]:<{max_col1}} = {col2[i]:{OUTPUT_FORMATTING_NUMBER}}\n"
            else:
                self.results_string += f"{col1[i]}\n"
                for index, string in enumerate(("Not optimized", "optimized")):
                    self.results_string += f"    {string:<{max_col1 - 4}} = {col2[i][index]:{OUTPUT_FORMATTING_NUMBER}}"
                    self.results_string += "\n"
        self.results_string += "*" * (max_col1 + 15) + '\n'
        if print_result:
            print(self.results_string)

        # combined_result_string
        columns = ['n', 'e_n(1)', 'e_n(2)', 'e_n(3)', 'e_n(4)', 't_tilde', 'hopp1', 'hopp2', 'epsil', 'Ec', 'd_occ1',
                   'd_occ2', 'U0/t_tilde', 'Occ_cluster', 'dsty1', 'dsty2', 'mu_KS', 'mu_imp', 'mu_ext']
        values = [n, self.e_site['without_mu_opt'], self.e_site['main'], self.e_site['type3'], self.e_site['type4'],
                  self.vars['t_tilde']] + self.vars['hopping'] + [self.vars['epsilon'], self.vars['KE']] + \
                 self.vars['d_occ'] + [self.U / self.vars['t_tilde'], self.gamma_tilde[0, 0] +
                                       self.gamma_tilde[1, 1]] + \
                 self.vars['density'] + [self.mu['KS'], self.mu['imp'], self.mu['ext']]
        val_str = ''.join([f'{num:{OUTPUT_FORMATTING_NUMBER}}' for num in values])
        self.combined_results_string['row'] = val_str + '\n'
        self.combined_results_string['col_names'] = "".join(
            [f'{i:>{int(len(val_str) / len(columns))}}' for i in columns])
        self.combined_results_string['col_names'] += '\n'
        # I put col names and row data into combined_results_string. I used such formatting that columns are aligned


def calculate_many_conditions(particle_number, U):
    data_string = ''
    result_string = ''
    for i in np.arange(2, particle_number * 2, 4):
        obj = Householder(particle_number, i, U)
        obj.calculate_one()
        if result_string == '':
            result_string = obj.combined_results_string['col_names']
        result_string += obj.combined_results_string['row']
        data_string += obj.results_string
    data_file = open(f"Single_loop-data-U_{U:.0f}-n.txt", 'w', encoding='UTF-8')
    data_file.write(data_string)
    data_file.close()
    result_file = open(f"Single_loop-results-U_{U:.0f}-n.txt", 'w', encoding='UTF-8')
    result_file.write(result_string)
    result_file.close()


if __name__ == "__main__":
    """obj = Householder(10, 4, 8)
    obj.calculate_one()
    print(obj.results_string)"""
    # print(obj.procedure_log)
    # calculate_many_conditions(100, 8)
    # calculate_many_conditions(10, 8)
    obj = Householder(10, 18, 8, debug=True)
    obj.calculate_one()
