import sys
import os
import typing

sys.path.append('../')
sys.path.append(r'C:\Users\tinc9\Documents\CNRS-offline')

import lpfet
import numpy as np
import matplotlib.pyplot as plt

import essentials
import seaborn as sns
import matplotlib.colors as mcolors

import importlib

importlib.reload(lpfet)

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_chain1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = []
    for j in range(n_sites):
        nodes_dict[j] = {'v': (j - 2.5) * i, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
        eq_list.append([j])
    return nodes_dict, edges_dict, eq_list


def generate_chain2(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = []
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
        eq_list.append([j])
    nodes_dict[0]['v'] = i
    return nodes_dict, edges_dict, eq_list


def generate_chain3(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = []
    for j in range(n_sites):
        nodes_dict[j] = {'v': i * (-1) ** j, 'U': U_param}
        if j != 5:
            edges_dict[(j, j + 1)] = 1
        eq_list.append([j])
    return nodes_dict, edges_dict, eq_list


def generate_star1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = [[0], list(range(1, n_sites))]
    for j in range(n_sites):
        nodes_dict[j] = {'v': i, 'U': U_param}
        if j != 0:
            edges_dict[(0, j)] = 1
    nodes_dict[0]['v'] = 0

    return nodes_dict, edges_dict, eq_list


def generate_complete1(i, n_sites, U_param):
    nodes_dict = dict()
    edges_dict = dict()
    eq_list = [[0]]
    stop1 = False
    for j in range(n_sites):
        nodes_dict[j] = {'v': 0, 'U': U_param}
        for k in range(j + 1, n_sites):
            edges_dict[(j, k)] = 1
        if j != 0 and not stop1:
            if j != n_sites - j:
                eq_list.append([j, n_sites - j])
            else:
                eq_list.append([j])
                stop1 = True
    nodes_dict[0]['v'] = i

    return nodes_dict, edges_dict, eq_list


def generate_trend(n_sites, n_electron, model_function: typing.Callable, molecule_name, u_param=None, i_param=None,
                   delta_x=None, max_value=None):
    lpfet.COMPENSATION_5_FACTOR = 1
    lpfet.COMPENSATION_5_FACTOR2 = 1
    lpfet.COMPENSATION_1_RATIO = 0.25
    name = molecule_name
    if u_param is not None and i_param is not None:
        raise Exception("Can't specify both parameters")
    elif u_param is not None:
        # U is set
        if delta_x is None:
            delta_x = 0.1
            max_value = 2
        x = np.arange(0, max_value + 0.01, delta_x)
        folder_name = f'results/i_{name}_ns-{n_sites}_ne-{n_electron}_u-{u_param}_dist-{delta_x}/'
        constant_var = 'u'

    elif i_param is not None:
        # i is set
        if delta_x is None:
            delta_x = 0.5
            max_value = 10
        x = np.arange(0, max_value + 0.01, delta_x)
        folder_name = f'results/u_{name}_ns-{n_sites}_ne-{n_electron}_i-{i_param}_dist-{delta_x}/'
        constant_var = 'i'
    else:
        raise Exception("None of the parameteres is set!")

    for one_path in [folder_name, folder_name + 'v_hxc', folder_name + 'occupation']:
        if not os.path.isdir(one_path):
            os.mkdir(one_path)
    mol1 = lpfet.Molecule(n_sites, n_electron, name)
    mol_full = lpfet.class_Quant_NBody.QuantNBody(n_sites, n_electron)
    mol_full.build_operator_a_dagger_a()
    y = []
    v_hxc_progression_list = []
    y_simple = []
    y_ref = []
    energy_ref = []
    energy = []
    first = True
    energy_per_site = []
    energy_ref_per_site = []
    correction_dict_list = []
    old_v_hxc = np.zeros(n_sites)
    with open(f"{folder_name}systems.txt", "w") as myfile:
        myfile.write("")
    for i in x:
        if constant_var == "i":
            u_param = i
        elif constant_var == "u":
            i_param = i
        if not first:
            mol1.clear_object(name)
            mol1.v_hxc = old_v_hxc
        first = False
        print(f'\n\n{i:.1f}, {i / max(x) * 100:.1f}%: ', end=' ')
        nodes_dict, edges_dict, eq_list = model_function(i_param, n_sites, u_param)
        t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
        mol1.report_string = f'Object with {n_sites} sites and {n_electron} electrons\n'
        with open(f"{folder_name}systems.txt", "a") as myfile:
            myfile.write(f"\n\n{repr(t)}\n{repr(v_ext)}\n{repr(u)}")
        mol1.add_parameters(u, t, v_ext, eq_list)
        mol1.self_consistent_loop(num_iter=50, tolerance=1E-6, oscillation_compensation=[5, 1],
                                  overwrite_output=folder_name + 'log.txt')
        # mol1.optimize_solution(5, 0.2)
        mol1.calculate_energy(True)
        correction_dict_list.append(mol1.oscillation_correction_dict)
        y.append(mol1.density_progress)
        y_simple.append(mol1.n_ks)
        y_ab, mol_fci, energy_ref_i, energy_ref_per_site_i = mol1.compare_densities_FCI(pass_object=mol_full,
                                                                                        calculate_per_site=True)
        energy_per_site.append(mol1.per_site_energy)
        energy_ref_per_site.append(energy_ref_per_site_i)
        y_ref.append(y_ab.diagonal())
        v_hxc_progression_list.append(mol1.v_hxc_progress)
        energy.append(mol1.energy_contributions)
        energy_ref.append(energy_ref_i)
        old_v_hxc = mol1.v_hxc.copy()

    calculate_graphs(folder_name, x, y, y_ref, y_simple, energy, energy_ref, v_hxc_progression_list,
                     correction_dict_list, energy_per_site, energy_ref_per_site)


def calculate_graphs(folder_name, x, y, y_ref, y_simple, energy, energy_ref, v_hxc_progression_list,
                     correction_dict_list, energy_per_site, energy_ref_per_site,
                     x_label='i (v_ext)'):
    for i in range(len(y)):
        y[i] = np.array(y[i], dtype=float)
    y_ref = np.array(y_ref)
    y_simple = np.array(y_simple)

    E = np.array(energy, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float)])
    E_ref = np.array(energy_ref, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float)])
    E_per_site = np.array(energy_per_site)
    E_ref_per_site = np.array(energy_ref_per_site)

    # saving variables
    np.savetxt(folder_name + 'x.dat', x)
    np.savetxt(folder_name + 'y_ref.dat', y_ref)
    np.savetxt(folder_name + 'y_simple.dat', y_simple)
    with open(folder_name + 'y.dat', 'w', encoding='UTF-8') as conn:
        conn.write(repr(y))
    with open(folder_name + 'v_hxc_progression_list.dat', 'w', encoding='UTF-8') as conn:
        conn.write(repr(v_hxc_progression_list))
    np.savetxt(folder_name + 'E.dat', E)
    np.savetxt(folder_name + 'E_ref.dat', E_ref)
    np.savetxt(folder_name + 'E_per_site.dat', E_per_site.view(float).reshape(E_per_site.shape[0], -1))
    np.savetxt(folder_name + 'E_ref_per_site.dat', E_ref_per_site.view(float).reshape(E_ref_per_site.shape[0], -1))
    # Because np.savetxt can be done only on 2d array we reshape array into 2d array that has each row
    # [tot1, kin1, v_ext1, u1, tot2, kin2, ..... , v_extn, un]

    # plot v_hxc progression
    for x_i, v_ext in enumerate(x):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        regime = np.array(v_hxc_progression_list[x_i])
        for site_id in range(6):
            plt.plot(np.arange(len(regime)) + 1, regime[:, site_id], color=mpl.cm.tab10(site_id),
                     label=f'site{site_id}')

        plt.xlabel("Iteration number")
        plt.ylabel("v_hxc")
        plt.title(f'v_xhc progression at regime with {x_label}={v_ext:.3f}')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'{folder_name}/v_hxc/progression-{x_i}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{folder_name}/v_hxc/progression-{x_i}.svg', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # plot occupation progression
    for x_i, v_ext in enumerate(x):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        regime = np.array(y[x_i])
        for site_id in range(6):
            plt.plot(np.arange(len(regime)) + 1, regime[:, site_id], color=mpl.cm.tab10(site_id),
                     label=f'site{site_id}')

        for key1 in correction_dict_list[x_i].keys():
            iter_key, site_key = key1
            plt.scatter([iter_key + 1], [regime[iter_key, site_key]], c='r', s=20)

        plt.xlabel("Iteration number")
        plt.ylabel("occupation")
        plt.title(f'occupation progression at regime with {x_label}={v_ext:.3f}')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 1)
        plt.savefig(f'{folder_name}/occupation/progression_{x_i}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{folder_name}/occupation/progression_{x_i}.svg', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # plot energy contributions
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, top=0.93)
    ax[0].grid(True, color='#E2E2E2')
    ax[1].grid(True, color='#E2E2E2')
    ax[0].plot(x, E['tot'], c=mpl.cm.get_cmap('tab10')(0), label='tot E')
    ax[0].plot(x, E_ref['tot'], c=mpl.cm.get_cmap('tab10')(0), linestyle='--')
    ax[0].plot(x, E['kin'], c=mpl.cm.get_cmap('tab10')(1), label='Kinetic contribution')
    ax[0].plot(x, E_ref['kin'], c=mpl.cm.get_cmap('tab10')(1), linestyle='--')
    ax[0].plot(x, E['u'], c=mpl.cm.get_cmap('tab10')(2), label='On-site repulsion')
    ax[0].plot(x, E_ref['u'], c=mpl.cm.get_cmap('tab10')(2), linestyle='--')
    ax[0].plot(x, E['v_ext'], c=mpl.cm.get_cmap('tab10')(3), label='Ext potential contribution')
    ax[0].plot(x, E_ref['v_ext'], c=mpl.cm.get_cmap('tab10')(3), linestyle='--')
    plt.xlabel(x_label)
    ax[0].set_ylabel('energy')
    ax[1].set_ylabel('Error energy')
    ax[1].plot(x, E['tot'] - E_ref['tot'], c=mpl.cm.get_cmap('tab10')(0))
    ax[1].plot(x, E['kin'] - E_ref['kin'], c=mpl.cm.get_cmap('tab10')(1))
    ax[1].plot(x, E['u'] - E_ref['u'], c=mpl.cm.get_cmap('tab10')(2))
    ax[1].plot(x, E['v_ext'] - E_ref['v_ext'], c=mpl.cm.get_cmap('tab10')(3))
    ax[1].plot(x, [0] * len(x), c='k', linestyle='--')
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{folder_name}/Energy_errors.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{folder_name}/Energy_errors.svg', dpi=300, bbox_inches='tight')

    # plot occupations
    colors = plt.cm.viridis(np.linspace(0, max(x), len(x)))
    norm = mpl.colors.Normalize(vmin=0, vmax=max(x))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for i in range(6):
        plt.plot(x, y_simple[:, i], c=mpl.cm.get_cmap('tab10')(i), label=str(i))
        plt.plot(x, y_ref[:, i], c=mpl.cm.get_cmap('tab10')(i), label=str(i) + '-ref', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel("occupation")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{folder_name}/Densities.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{folder_name}/Densities.svg', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.tight_layout()
    plt.xlabel(x_label)
    ax.set_ylabel('Error energy')
    ax.set_ylim(0, 1)
    for site in range(6):
        ax.plot(x, E_per_site[:, site]['tot'] - E_ref_per_site[:, site]['tot'], c=mpl.cm.get_cmap('tab10')(site),
                label=f'deviation on site {site}')
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(f'{folder_name}/Energy_errors_per_site.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{folder_name}/Energy_errors_per_site.svg', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # generate_trend(6, 6, generate_chain2, 'chain2', u_param=5)
    # generate_trend(6, 6, generate_chain3, 'chain3', u_param=5, delta_x=0.4, max_value=4)
    # generate_trend(6, 6, generate_star1, 'star1', u_param=5, delta_x=0.4, max_value=4)
    # generate_trend(6, 6, generate_complete1, 'complete1', u_param=5, delta_x=0.1, max_value=3)
    mol1 = lpfet.Molecule(6, 6, 'hehe')
    nodes_dict, edges_dict, eq_list = generate_chain1(2, 6, 10)
    t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
    mol1.add_parameters(u, t, v_ext, eq_list)
    mol1.self_consistent_loop(50, 1e-4, oscillation_compensation=[5, 1])
    mol_full = lpfet.class_Quant_NBody.QuantNBody(6, 6)
    mol_full.build_operator_a_dagger_a()
    U = np.zeros((6, 6, 6, 6))
    for i in range(6):
        U[i, i, i, i] = u[i]
    mol_full.build_hamiltonian_fermi_hubbard(t+np.diag(v_ext), U)
    mol_full.diagonalize_hamiltonian()
    tuple1 = mol_full.calculate_v_hxc(mol1.v_hxc)
    pass
