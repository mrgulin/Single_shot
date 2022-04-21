from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
from LPFET.calculate_batch import generate_trend
import LPFET.lpfet as lpfet
import LPFET.essentials as essentials


def chain1_cont_i():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_i in [0.5, 1, 1.5, 2]:
            input_list.append((6, electrons, essentials.generate_chain1, 'chain1', None, param_i, None, None, None))
    with Pool(processes=10) as pool:
        pool.starmap(generate_trend, input_list)


def ring5_cont_i():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_i in [-1, 0, 0.5, 1, 1.5]:
            input_list.append((6, electrons, essentials.generate_ring5, 'ring5', None, param_i, None, None, None))
    with Pool(processes=10) as pool:
        pool.starmap(generate_trend, input_list)


def complete1_cont_i():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_i in [-1, 0, 0.5, 1, 1.5]:
            input_list.append((6, electrons, essentials.generate_complete1, 'complete1', None, param_i, None, None,
                               None, False))
    with Pool(processes=10) as pool:
        pool.starmap(generate_trend, input_list)


def star1_cont_i():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_i in [-1, 0, 0.5, 1, 1.5]:
            input_list.append(
                (6, electrons, essentials.generate_star1, 'star1', None, param_i, None, None,
                 None, False))
    with Pool(processes=10) as pool:
        pool.starmap(generate_trend, input_list)


def chain1_cont_u():
    input_list = []
    for electrons in [2, 4, 6, 8, 10]:
        for param_u in [2, 4, 6, 8, 10]:
            input_list.append((6, electrons, essentials.generate_chain1, 'chain1', param_u, None, None, None, None))
        with Pool(processes=10) as pool:
            pool.starmap(generate_trend, input_list)


def chain1_cont_i_with_nni():
    input_list = []
    for r in [0.1, 0.2, 0.4]:
        for electrons in [2, 4, 6, 8, 10]:
            for param_i in [0.5, 1, 1.5, 2]:
                input_list.append((6, electrons, essentials.generate_chain1, f'chain1_NNI-{r:.1f}', None, param_i, None,
                                   None, r))
    with Pool(processes=10) as pool:
        pool.starmap(generate_trend, input_list)


if __name__ == "__main__":
    freeze_support()
    # chain1_cont_i()
    # chain1_cont_u()
    # chain1_cont_i_with_nni()
    # ring5_cont_i()
    star1_cont_i()
    complete1_cont_i()
