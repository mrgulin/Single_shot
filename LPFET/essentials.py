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
    return y * 2


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


def generate_chain1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': (j - 2.5) * i, 'U': U_param}
        if j != n_sites - 1:
            edges_dict[(j, j + 1)] = 1
    return nodes_dict, edges_dict


def generate_chain2(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
    nodes_dict[0]['v'] = i
    return nodes_dict, edges_dict


def generate_chain3(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': i * (-1) ** j, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
    return nodes_dict, edges_dict


def generate_star1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': i, 'U': U_param}
        if j != 0:
            edges_dict[(0, j)] = 1
    nodes_dict[0]['v'] = 0

    return nodes_dict, edges_dict


def generate_complete1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        for k in range(j + 1, n_sites):
            edges_dict[(j, k)] = 1
    nodes_dict[0]['v'] = i
    return nodes_dict, edges_dict

def generate_ring4(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        if j != n_sites - 1:
            edges_dict[(j, j + 1)] = 1
        else:
            edges_dict[(0, j)] = 1
    nodes_dict[0]['v'] = i
    nodes_dict[1]['v'] = i
    return nodes_dict, edges_dict

