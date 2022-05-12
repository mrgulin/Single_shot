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
    """
    -2.5i --- -1.5i --- -0.5i --- ... --- (-2.5 + n_sites) * i
    :param i: difference between levels
    :param n_sites: Length of the chain
    :param U_param: U
    :return: classic output for generate functions
    """
    nodes_dict = dict()
    edges_dict = dict()
    minimum = (n_sites - 1)/2
    for j in range(n_sites):
        nodes_dict[j] = {'v': (j - minimum) * i, 'U': U_param}
        if j != n_sites - 1:
            edges_dict[(j, j + 1)] = 1
    return nodes_dict, edges_dict


def generate_chain2(i, n_sites, U_param):
    """
    i --- 0 --- 0 --- ... --- 0
    :param i: difference between levels
    :param n_sites: Length of the chain
    :param U_param: U
    :return: classic output for generate functions
    """
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
    nodes_dict[0]['v'] = i
    return nodes_dict, edges_dict


def generate_chain3(i, n_sites, U_param):
    """
    -i --- +i --- -i --- ... --- ((-1) ** n_sites) i
    :param i: difference between levels
    :param n_sites: Length of the chain
    :param U_param: U
    :return: classic output for generate functions
    """
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
    """
    Ring with ext potential:
        i - i
      /      ＼
     0        0
      ＼     /
       0 - 0
    :param i: external potential on site 0 and 1
    :param n_sites: number of sites
    :param U_param: U parameter
    :return: usual return from generate ...
    """
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


def generate_ring5(i, n_sites, U_param):
    """
    Something similar to cyclopentanol. i parameter is on substituent
    0 - 0
    |    ＼
    |     0 - i
    |    /
    0 - 0
    :param i: external potential on site 0 and 1
    :param n_sites: number of sites
    :param U_param: U parameter
    :return: usual return from generate ...
    """
    nodes_dict = dict()
    edges_dict = dict()
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        if j != n_sites - 1:
            edges_dict[(j, j + 1)] = 1
        else:
            edges_dict[(1, j)] = 1
    nodes_dict[0]['v'] = i
    return nodes_dict, edges_dict


def generate_random1(i, n_sites, U_param):
    # for u param: np.random.random(8) ** 2 * 1.5 + 0.5
    # [1.60284607, 0.72620286, 0.51503746, 1.18798946, 0.52910353,
    #  1.18323482, 1.758816  , 0.93973849]

    # for t: np.random.random(9)  * 1.0 + 0.5
    # [1.01605887 1.33261473 1.01268904 0.86255098 1.40662862 0.79671075
    #  1.49033559 0.77958568 1.09682106]

    # for v_ext: np.random.random(8)  * 4 - 2
    # [0.79104355  0.14024618 - 1.67135683 - 0.42730853  1.3236309   1.22639956
    #  - 0.67630773 - 1.09101013]
    ret = ({0: {'v': +0.79 * i, 'U': 1.60 * U_param},
            1: {'v': +0.14 * i, 'U': 0.73 * U_param},
            2: {'v': -1.67 * i, 'U': 0.52 * U_param},
            3: {'v': -0.43 * i, 'U': 1.19 * U_param},
            4: {'v': +1.32 * i, 'U': 0.53 * U_param},
            5: {'v': +1.23 * i, 'U': 1.18 * U_param},
            6: {'v': -0.68 * i, 'U': 1.75 * U_param},
            7: {'v': -1.09 * i, 'U': 0.94 * U_param}
            },
           {(0, 1): 1.02, (1, 2): 1.33, (2, 3): 1.01, (3, 4): 0.86, (4, 5): 1.41, (5, 6): 0.80, (0, 6): 1.49,
            (0, 2): 0.78, (5, 7): 1.10})
    return ret
