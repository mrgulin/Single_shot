import numpy as np
import matplotlib.pyplot as plt

def print_matrix(matrix, decimal_num="+.2f", separator="  ", plot_heatmap=True, ret=False):
    ret_string = ""
    for line in matrix:
        for cell in line:
            ret_string += '{num:{dec}}{separator}'.format(num=cell, field_size=decimal_num, separator=separator)
        ret_string = ret_string[:-len(separator)]
        ret_string += "\n"
    if plot_heatmap:
        plt.imshow(obj.gamma, cmap='hot', interpolation='nearest')
        plt.show()
    if ret:
        return ret_string
    print(ret_string, end='')

class Householder:
    def __init__(self, particle_number: int, u: float):
        self.N = particle_number

        self.t = 1.00
        self.U = u
        self.gamma = np.zeros((self.N, self.N), dtype=np.float64)
        self.gamma_tilde = np.zeros((self.N, self.N), dtype=np.float64)
        self.h = np.zeros((self.N, self.N), dtype=np.float64)  # non-interacting hamiltonian h_ij

        self.P = np.zeros((self.N, self.N), dtype=np.float64)  # Householder transformation matrix
        self.v = np.zeros((self.N), dtype=np.float64)  # Householder vector v

        self.procedure_log = ""
    def calculate_one(self, Ne):
        """
        Equivalent to DENSITY_MATRIX subroutine
        :Ne: Number of electrons for the calculation
        """
        if (Ne % 2) != 0 or type(Ne) != int:
            raise 'Problem! Number of electrons is not even!'

        n = float(Ne) / float(self.N)  # Density

        # Huckel hamiltonian generation: self.h in our case
        self.h = np.zeros((self.N, self.N), dtype=np.float64)  # reinitialization

        if (Ne / 2) % 2 != 0:
            self.h[0, self.N - 1] = self.t
            self.h[self.N - 1, 0] = self.t
        else:
            self.h[0, self.N - 1] = -self.t
            self.h[self.N - 1, 0] = -self.t

        self.h += np.diag(np.full((self.N - 1), -self.t), -1) + np.diag(np.full((self.N - 1), -self.t), 1)

        ei_val, ei_vec = np.linalg.eig(self.h)  # v[:,i] corresponds to eigval w[i]

        # SORTING THE EIGENVALUE!
        idx = ei_val.argsort()[::1]
        ei_val = ei_val[idx]
        ei_vec = ei_vec[:, idx]
        # generation of 1RDM
        self.gamma = np.zeros((self.N, self.N), dtype=np.float64)  # reset gamma
        # for Ne_cnt in range(0, Ne, 2): then we would have k goes from 0 to ...
        for k in range(int(Ne / 2)):  # go through all orbitals that are occupied!
            for i in range(self.N):
                for j in range(self.N):
                    self.gamma[i, j] += ei_vec[i, k] * ei_vec[j, k]
        # print_matrix(self.gamma)

        mu_KS = ei_val[Ne//2]
        # end of subroutine
        self.procedure_log += f"CALCULATIONS MADE FOR THE KS SYSTEM WITH Ns = {self.N} and Ne = {Ne}\n"
        self.procedure_log += f"DENSITY = {n}\n THE CHEMICAL POTENTIAL (mu_KS) ASSOCIATED WITH THE NUMBER OF ELECTRONS"
        self.procedure_log += f" = {mu_KS}\n GAMMA0 \n\n {print_matrix(self.gamma, '.3f', ',', False, True)}\n"

        # Householder vector generation


    def calculate_many_conditions(self):
        for i in np.arange(2, self.N * 2, 4):
            self.calculate_one(self)

    def generate_householder_vector(self):
        sum_M = 0
        # SHIFTED INDICES!!
        for j in range(1, self.N):
            sum_M += self.gamma[j, 0] * self.gamma[j, 0]
        alpha = -1 * np.sign(self.gamma[1,0]) * np.sqrt(sum_M)  # in notes it is xi
        r = np.sqrt(0.5 * alpha * (alpha - self.gamma[1, 0]))

        self.v = np.zeros((self.N, ), dtype=np.float64)  # reset array, v[0] = 0 so it is okay
        self.v[1] = (self.gamma[1,0] - alpha)/(2. * r)


if __name__ == "__main__":
    obj = Householder(16, 5)

    print(obj.calculate_one(8))
