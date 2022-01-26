import numpy as np

mat2 = np.array(
    [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
)
OUTPUT_FORMATTING_NUMBER = "+12.6f"
OUTPUT_SEPARATOR = "  "


def generate_huckel_hamiltonian(N=12, number_of_electrons=12, t=1):
    h = np.zeros((N, N), dtype=np.float64)  # reinitialization
    if (number_of_electrons / 2) % 2 == 0:
        h[0, N - 1] = t
        h[N - 1, 0] = t
    else:
        h[0, N - 1] = -t
        h[N - 1, 0] = -t

    h += np.diag(np.full((N - 1), -t), -1) + np.diag(np.full((N - 1), -t), 1)
    return h


mat2 = generate_huckel_hamiltonian()


def print_matrix(matrix, plot_heatmap='', ret=False):
    ret_string = ""
    for line in matrix:
        l1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in line]
        ret_string += f'{OUTPUT_SEPARATOR}'.join(l1) + "\n"
    if ret:
        return ret_string
    print(ret_string, end='')


ei_val, ei_vec = np.linalg.eigh(mat2)

print(ei_val)
print("----")
import single_shot as ss
ss.print_matrix(ei_vec, 'test2')
def calculate_gamma(ei_vec):
    N = ei_vec.shape[0]
    gamma = np.zeros((N, N), dtype=np.float64)
    for k in range(int(12 / 2)):  # go through all orbitals that are occupied!
        vec_i = ei_vec[:, k][np.newaxis]
        gamma += vec_i.T @ vec_i  #
        # for i in range(N):
        #     for j in range(N):
        #         gamma[i, j] += ei_vec[i, k] * ei_vec[j, k]
    return gamma
print(11111111)
print_matrix(calculate_gamma(np.array(ei_vec,dtype=np.float64)))
