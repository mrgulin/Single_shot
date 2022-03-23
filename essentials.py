import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FORMATTING_NUMBER = "+8.3f"
OUTPUT_SEPARATOR = " "


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


def generate_1rdm(Ns, Ne, wave_function):
    # generation of 1RDM
    if Ne % 2 != 0:
        raise f'problem with number of electrons!! Ne = {Ne}, Ns = {Ns}'

    y = np.zeros((Ns, Ns), dtype=np.float64)  # reset gamma
    for k in range(int(Ne / 2)):  # go through all orbitals that are occupied!
        vec_i = wave_function[:, k][np.newaxis]
        y += vec_i.T @ vec_i  #
    return y


def generate_huckel_hamiltonian(n_s, number_of_electrons=2, t=1):
    h = np.zeros((n_s, n_s), dtype=np.float64)  # reinitialization
    if (number_of_electrons / 2) % 2 == 0:
        h[0, n_s - 1] = t
        h[n_s - 1, 0] = t
        pl = "ANTIPERIODIC" + '\n'
    else:
        h[0, n_s - 1] = -t
        h[n_s - 1, 0] = -t
        pl = "PERIODIC" + '\n'

    h += np.diag(np.full((n_s - 1), -t), -1) + np.diag(np.full((n_s - 1), -t), 1)
    print(pl)
    return h
