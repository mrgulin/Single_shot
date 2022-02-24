import os
os.chdir(r"C:\Users\tinc9\Documents\CNRS-offline\Single_shot")
import Self_consistent_loop_Saad_code as Scl

meta_obj = Scl.MetaScHouseholder(10, 1, -10, 10, 0.1)
meta_obj.calculate_many_potentials()

import numpy as np
import matplotlib.pyplot as plt


def data(keyword, array, header):
    return array[:, header.index(keyword)]


def plot_data(kw_x, kw_y, array, header):
    ymin = min(array[:, header.index(kw_y)])
    ymax = max(array[:, header.index(kw_y)])
    plt.vlines(x=[-2., -1., -1., 1., 1., 2.], ymin=ymin, ymax=ymax)
    plt.scatter(array[:, header.index(kw_x)], array[:, header.index(kw_y)], marker='x')
    plt.xlabel(kw_x)
    plt.ylabel(kw_y)
    plt.show()

def draw_graphs(path_results):
    array = np.loadtxt(path_results, skiprows=1)
    header = ['N', 'Ne', 'density', 'mu_ks', 'mu_hxc', 'mu_ext', 'mu_xc', 'iter_num', 'site_e']


    # plot_data('mu_ext', 'Ne')
    # plot_data('mu_ext', 'mu_ks')
    # plot_data('mu_ext', 'mu_hxc')
    # plot_data('mu_ext', 'mu_xc')

    # Graph for Ne dependence on mu_ext
    plt.scatter(array[:, header.index("mu_ext")], array[:, header.index('Ne')], marker='x',
                c=array[:, header.index('mu_ks')])
    plt.colorbar()
    plt.xlabel('mu_ext')
    plt.ylabel("Ne")
    plt.title(path_results)
    plt.savefig(path_results[:-4]+"_Ne_color.png")
    plt.show()

    plt.scatter(array[:, header.index("mu_ext")], array[:, header.index('Ne')], marker='x',
                c='green', label='Ne')
    plt.scatter(array[:, header.index("mu_ext")], array[:, header.index('mu_ks')], marker='x',
                c='red', label='mu_ks')
    plt.scatter(array[:, header.index("mu_ext")], array[:, header.index('mu_hxc')], marker='x',
                c='black', label='mu_hxc')
    plt.xlabel('mu_ext')
    plt.legend()
    plt.title(path_results)
    plt.savefig(path_results[:-4]+"_multiple_graphs.png")
    plt.show()
# draw_graphs('results_sc/Saad_N-10_U-4_201.dat')
# draw_graphs('results_sc/Saad_N-10_U-1_201.dat')
# draw_graphs('results_sc/output2.dat')
draw_graphs('results_sc/output2_N-10_U-1.dat')


# Comparison between py and fr:
# array_fr = np.loadtxt('results_sc/output2-N-10_U-4.dat', skiprows=1)
# array_py = np.loadtxt('results_sc/Saad_N-10_U-4_201.dat', skiprows=1)
# plt.scatter(array_py[:, 5], array_py[:, 1], marker='x',
#             c='green', label='Ne')
# plt.scatter(array_fr[:, 5], array_fr[:, 1]+0.05, marker='x',
#             c='k', label='Ne')
# plt.show()