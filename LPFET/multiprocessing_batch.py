from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

import calculate_batch


def chain1_cont_i():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_i in [0.5, 1, 1.5, 2]:
            input_list.append((param_i, electrons, 6, 0.5))
    with Pool() as pool:
        L = pool.starmap(calculate_batch.generate_chain_const_i, input_list)

def chain1_cont_u():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_u in [2, 4, 6, 8, 10]:
            input_list.append((param_u, electrons, 6))
    with Pool() as pool:
        L = pool.starmap(calculate_batch.generate_chain_const_u, input_list)


if __name__ == "__main__":
    freeze_support()
    # chain1_cont_i()
    chain1_cont_u()
