import Quant_NBody
import numpy as np
import scipy.sparse

OUTPUT_FORMATTING_NUMBER = "+15.10f"
OUTPUT_SEPARATOR = "  "
import matplotlib.pyplot as plt
from matplotlib import cm


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


class QuantNBody:
    def __init__(self, N_MO, N_elec, S_z_cleaning=False, override_NBody_basis=tuple()):
        self.N_MO = N_MO
        self.N_elec = N_elec
        if not override_NBody_basis:
            self.NBody_Basis = Quant_NBody.Build_NBody_Basis(N_MO, N_elec, S_z_cleaning)
        else:
            self.NBody_Basis = override_NBody_basis
        self.a_dagger_a = []

        self.H = scipy.sparse.csr_matrix((1, 1))
        self.h = np.array([], dtype=np.float64)
        self.U = np.array([], dtype=np.float64)
        self.ei_val = self.ei_vec = np.array([])
        self.WFT_0 = np.array([])

    def build_operator_a_dagger_a(self):
        self.a_dagger_a = Quant_NBody.Build_operator_a_dagger_a(self.NBody_Basis)

    def build_hamiltonian_quantum_chemistry(self, h_, U_, *args, **kwargs):
        self.h = h_
        self.U = U_
        self.H = Quant_NBody.Build_Hamiltonian_Quantum_Chemistry(h_, U_, self.NBody_Basis, self.a_dagger_a,
                                                                 *args, **kwargs)

    def build_hamiltonian_fermi_hubbard(self, h_, U_, *args, **kwargs):
        self.h = h_
        self.U = U_
        self.H = Quant_NBody.Build_Hamiltonian_Fermi_Hubbard(h_, U_, self.NBody_Basis, self.a_dagger_a,
                                                             *args, **kwargs)

    def diagonalize_hamiltonian(self):
        if len(self.H.A) == 0:
            print('You have to generate H first')

        self.ei_val, self.ei_vec = np.linalg.eigh(self.H.A)
        self.WFT_0 = self.ei_vec[:, 0]

    def visualize_coefficients(self, index, cutoff=0.005):
        # Quant_NBody.Visualize_WFT(self.ei_vec[index], self.NBody_Basis, cutoff=cutoff)
        WFT = self.ei_vec[:, index]
        list_indx = np.where(abs(WFT) > cutoff)[0]

        States = []
        Coeffs = []
        for indx in list_indx:
            Coeffs += [WFT[indx]]
            States += [self.NBody_Basis[indx]]

        list_sorted_indx = np.flip(np.argsort(np.abs(Coeffs)))

        print()
        print('\t ----------- ')
        print('\t Coeff.      N-body state')
        print('\t -------     -------------')
        for indx in list_sorted_indx[0:8]:
            sign_ = '+'
            if (abs(Coeffs[indx]) != Coeffs[indx]): sign_ = '-'
            print('\t', sign_, '{:1.5f}'.format(abs(Coeffs[indx])),
                  '\t' + get_better_ket(States[indx]))

    def calculate_1RDM_tot(self, index=0):
        return Quant_NBody.Build_One_RDM_spin_free(self.ei_vec[:, index], self.a_dagger_a)


def get_better_ket(state, bra=False):
    ret_string = ""
    for i in range(len(state) // 2):
        if state[i * 2] == 1:
            if state[i * 2 + 1] == 1:
                ret_string += '2'
            else:
                ret_string += 'a'
        elif state[i * 2 + 1] == 1:
            ret_string += 'b'
        else:
            ret_string += '0'
    if bra:
        ret_string = '⟨' + ret_string + '|'
    else:
        ret_string = '|' + ret_string + '⟩'
    return ret_string


def build_chain(N_MO, alpha=1):
    t = np.zeros((N_MO, N_MO))
    U = np.zeros((N_MO, N_MO, N_MO, N_MO))
    for i in range(N_MO):
        U[i, i, i, i] = i ** 2 * alpha  # Local coulombic repulsion
    for i in range(N_MO - 1):
        t[i, i + 1] = t[i + 1, i] = - 1  # hopping
    h = t
    return h, U

def build_chain(N_MO, alpha=1):
    t = np.zeros((N_MO, N_MO))
    U = np.zeros((N_MO, N_MO, N_MO, N_MO))
    for i in range(N_MO):
        U[i, i, i, i] = i ** 2 * alpha  # Local coulombic repulsion
    for i in range(N_MO - 1):
        t[i, i + 1] = t[i + 1, i] = - 1  # hopping
    h = t
    return h, U

def build_complete_graph(N_MO, alpha=1):
    U = np.zeros((N_MO, N_MO, N_MO, N_MO))
    t = np.zeros((N_MO, N_MO))
    t -= -1
    for i in range(N_MO):
        t[i, i] = 0  # hopping
    for i in range(N_MO):
        U[i, i, i, i] = i ** 2 * alpha  # Local coulombic repulsion
    h = t
    return h, U

def build_ring(N_MO, alpha=1, type_of='even_odd'):
    U = np.zeros((N_MO, N_MO, N_MO, N_MO))
    h = np.zeros((N_MO, N_MO))
    for i in range(N_MO):
        h[i, i] = 0  # hopping
    for i in range(N_MO):
        if type_of == 'even_odd':
            U[i, i, i, i] = (1 + i % 2) * alpha  # Local coulombic repulsion

    if type_of == 'opposite':
        U[0, 0, 0, 0] = alpha
        U[N_MO//2, N_MO//2, N_MO//2, N_MO//2] = alpha

    if (N_MO / 2) % 2 == 0:
        h[0, N_MO - 1] = 1
        h[N_MO - 1, 0] = 1
        pl = "ANTIPERIODIC" + '\n'
    else:
        h[0, N_MO - 1] = -1
        h[N_MO - 1, 0] = -1
        pl = "PERIODIC" + '\n'

    h += np.diag(np.full((N_MO - 1), -1), -1) + np.diag(np.full((N_MO - 1), -1), 1)
    return h, U


def generate_pos(N_MO, h_u_generator, alpha_lim=(0.1, 5, 1)):
    obj = QuantNBody(N_MO, N_MO)
    obj.build_operator_a_dagger_a()
    # saved_NBody_Basis = obj.NBody_Basis
    # save_a_dagger_a = obj.a_dagger_a
    energy_list = []
    i = 0
    tot_i = len(np.arange(*alpha_lim))
    density_list = []
    for alpha in np.arange(*alpha_lim):
        print(f'{i+1}/{tot_i}')
        obj.build_hamiltonian_fermi_hubbard(*h_u_generator(N_MO, alpha))
        obj.diagonalize_hamiltonian()
        densities = np.diagonal(obj.calculate_1RDM_tot(0))
        density_list.append(densities)
        energy_list.append([alpha, obj.ei_val[0]])
        i += 1
    energy_list = np.array(energy_list)
    plt.scatter(energy_list[:, 0], energy_list[:, 1])
    plt.xlabel('alpha in expression for U')
    plt.ylabel('Energy of ground state')
    plt.show()
    c = np.array([[alpha] * N_MO for alpha in np.arange(*alpha_lim)])
    x = np.array([range(N_MO) for alpha in np.arange(*alpha_lim)])
    y = np.array(density_list)
    for i in range(len(x)):
        plt.plot(x[i], y[i], c=cm.viridis(i/len(x)))
    plt.xlabel('site id')
    plt.ylabel('occupancy')
    plt.show()

if __name__ == '__main__':
    # generate_pos(6, build_chain, alpha_lim=(0.01, 0.2, 0.02))
    # generate_pos(6, build_complete_graph, alpha_lim=(0.001, 0.01, 0.002))
    # generate_pos(6, build_ring, alpha_lim=(0.1, 3, 0.5))
    generate_pos(6, build_chain, alpha_lim=(0.1, 3, 0.5))