import sys
import os
import typing
from datetime import datetime
from LPFET import lpfet
import numpy as np
import matplotlib.pyplot as plt
from LPFET import essentials
import matplotlib as mpl


def generate_trend(n_sites, n_electron, model_function: typing.Callable, molecule_name, u_param=None, i_param=None,
                   delta_x=None, max_value=None, r_param=None, force=True):
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
        raise Exception("None of the parameters is set!")

    for one_path in [folder_name, folder_name + 'v_hxc', folder_name + 'occupation']:
        if not os.path.isdir(one_path):
            os.mkdir(one_path)

    if not force:
        if os.path.isfile(folder_name + "/Energy_errors_per_site.svg"):
            print("THERE IS ALREADY DATA. FUNCTION WILL EXIT.")
            return 0

    mol1 = lpfet.Molecule(n_sites, n_electron, name)
    mol_full = lpfet.class_qnb.HamiltonianV2(n_sites, n_electron)
    mol_full.build_operator_a_dagger_a()
    y = []
    v_hxc_progression_list = []
    y_simple = []
    y_ref = []
    energy_ref = []
    energy = []
    v_hxc_ref_progress = []
    first = True
    energy_per_site = []
    energy_ref_per_site = []
    correction_dict_list = []
    old_v_hxc = None
    starting_approximation_c_hxc = None
    time_list = []
    approx_len = None
    with open(f"{folder_name}systems.txt", "w") as my_file:
        my_file.write("")
    x_label = ""
    for i in x:
        if constant_var == "i":
            u_param = i
            x_label = 'U'
        elif constant_var == "u":
            i_param = i
            x_label = 'i (delta v_ext)'
        if not first:
            approx_len = len(mol1.equiv_atom_groups) - 1
            starting_approximation_c_hxc = np.zeros(approx_len)
            for ind in range(approx_len):
                starting_approximation_c_hxc[ind] = old_v_hxc[mol1.equiv_atom_groups[ind + 1][0]]
            mol1.clear_object(name)
        print(f'\n\n{i:.1f}, {i / max(x) * 100:.1f}%: ', end=' ')
        nodes_dict, edges_dict = model_function(i_param, n_sites, u_param)
        t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
        mol1.report_string = f'Object with {n_sites} sites and {n_electron} electrons\n'
        with open(f"{folder_name}systems.txt", "a") as my_file:
            my_file.write(f"\n\n{repr(t)}\n{repr(v_ext)}\n{repr(u)}")
        mol1.add_parameters(u, t, v_ext, r_param)
        if len(mol1.equiv_atom_groups) - 1 != approx_len:
            starting_approximation_c_hxc = None
        time1 = datetime.now()
        # mol1.self_consistent_loop(num_iter=50, tolerance=1E-6, oscillation_compensation=[5, 1], v_hxc_0=0)
        try:
            mol1.find_solution_as_root(starting_approximation_c_hxc)
        except lpfet.errors.DegeneratedStatesError as e:
            print('Degenerated energy levels --> skipping this system.\n', e, '--end--\n')
            continue
        except lpfet.errors.HouseholderTransformationError as e:
            print('Unable to calculate Householder transformation --> skipping this system.\n', e, '--end--\n')
            continue
        time2 = datetime.now()
        # mol1.optimize_solution(5, 0.2)
        mol1.calculate_energy(False)
        time3 = datetime.now()
        first = False
        correction_dict_list.append(mol1.oscillation_correction_dict)
        y.append(mol1.density_progress)
        y_simple.append(mol1.n_ks)
        y_ab, mol_fci, energy_ref_i, energy_ref_per_site_i = mol1.compare_densities_fci(pass_object=mol_full,
                                                                                        calculate_per_site=True)
        time4 = datetime.now()
        v_hxc_correct = mol_fci.calculate_v_hxc(mol1.v_hxc)
        time5 = datetime.now()
        print(v_hxc_correct)
        if type(v_hxc_correct) == bool:
            print("didn't manage to converge")
            v_hxc_ref_progress.append(np.zeros(n_sites) * np.nan)
        else:
            v_hxc_ref_progress.append(v_hxc_correct.copy())
        energy_per_site.append(mol1.per_site_energy)
        energy_ref_per_site.append(energy_ref_per_site_i)
        y_ref.append(y_ab.diagonal())
        v_hxc_progression_list.append(mol1.v_hxc_progress)
        energy.append(mol1.energy_contributions)
        energy_ref.append(energy_ref_i)
        old_v_hxc = mol1.v_hxc.copy()
        time_list.append([(time2-time1).total_seconds(), (time3-time2).total_seconds(), (time4-time3).total_seconds(),
                          (time5-time4).total_seconds()])
    time_before_graphs = datetime.now()
    calculate_graphs(folder_name, x, y, y_ref, y_simple, energy, energy_ref, v_hxc_progression_list,
                     correction_dict_list, energy_per_site, energy_ref_per_site, v_hxc_ref_progress, x_label)
    print(f"time spent for making graphs: {(datetime.now()-time_before_graphs).total_seconds()}")
    return np.array(time_list)


# noinspection PyTypeChecker
def calculate_graphs(folder_name, x, y, y_ref, y_simple, energy, energy_ref, v_hxc_progression_list,
                     correction_dict_list, energy_per_site, energy_ref_per_site, v_hxc_ref_progress,
                     x_label='i (v_ext)'):
    for i in range(len(y)):
        y[i] = np.array(y[i], dtype=float)
    y_ref = np.array(y_ref)
    y_simple = np.array(y_simple)

    e = np.array(energy, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),  ('v_term', float)])
    e_ref = np.array(energy_ref, dtype=[('tot', float), ('kin', float), ('v_ext', float), ('u', float),
                                        ('v_term', float)])
    e_per_site = np.array(energy_per_site)
    e_ref_per_site = np.array(energy_ref_per_site)
    v_hxc_ref_progress = np.array(v_hxc_ref_progress)

    # saving variables
    np.savetxt(folder_name + 'x.dat', x)
    np.savetxt(folder_name + 'y_ref.dat', y_ref)
    np.savetxt(folder_name + 'y_simple.dat', y_simple)
    with open(folder_name + 'y.dat', 'w', encoding='UTF-8') as conn:
        conn.write(repr(y))
    with open(folder_name + 'v_hxc_progression_list.dat', 'w', encoding='UTF-8') as conn:
        conn.write(repr(v_hxc_progression_list))
    np.savetxt(folder_name + 'v_hxc_ref_progress.dat', v_hxc_ref_progress)
    np.savetxt(folder_name + 'e.dat', e)
    np.savetxt(folder_name + 'e_ref.dat', e_ref)
    np.savetxt(folder_name + 'e_per_site.dat', e_per_site.view(float).reshape(e_per_site.shape[0], -1))
    np.savetxt(folder_name + 'e_ref_per_site.dat', e_ref_per_site.view(float).reshape(e_ref_per_site.shape[0], -1))
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
        # plt.savefig(f'{folder_name}/v_hxc/progression-{x_i}.png', dpi=150, bbox_inches='tight')
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
        ax.set_ylim(0, 2)
        # plt.savefig(f'{folder_name}/occupation/progression_{x_i}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{folder_name}/occupation/progression_{x_i}.svg', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # plot energy contributions
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, top=0.93)
    ax[0].grid(True, color='#E2E2E2')
    ax[1].grid(True, color='#E2E2E2')
    ax[0].plot(x, e['tot'], c=mpl.cm.get_cmap('tab10')(0), label='tot e')
    ax[0].plot(x, e_ref['tot'], c=mpl.cm.get_cmap('tab10')(0), linestyle='--')
    ax[0].plot(x, e['kin'], c=mpl.cm.get_cmap('tab10')(1), label='Kinetic contribution')
    ax[0].plot(x, e_ref['kin'], c=mpl.cm.get_cmap('tab10')(1), linestyle='--')
    ax[0].plot(x, e['u'], c=mpl.cm.get_cmap('tab10')(2), label='On-site repulsion')
    ax[0].plot(x, e_ref['u'], c=mpl.cm.get_cmap('tab10')(2), linestyle='--')
    ax[0].plot(x, e['v_ext'], c=mpl.cm.get_cmap('tab10')(3), label='Ext potential contribution')
    ax[0].plot(x, e_ref['v_ext'], c=mpl.cm.get_cmap('tab10')(3), linestyle='--')
    ax[0].plot(x, e['v_term'], c=mpl.cm.get_cmap('tab10')(4), label='v_term contribution')
    ax[0].plot(x, e_ref['v_term'], c=mpl.cm.get_cmap('tab10')(4), linestyle='--')
    plt.xlabel(x_label)
    ax[0].set_ylabel('energy')
    ax[1].set_ylabel('Error energy')
    ax[1].plot(x, e['tot'] - e_ref['tot'], c=mpl.cm.get_cmap('tab10')(0))
    ax[1].plot(x, e['kin'] - e_ref['kin'], c=mpl.cm.get_cmap('tab10')(1))
    ax[1].plot(x, e['u'] - e_ref['u'], c=mpl.cm.get_cmap('tab10')(2))
    ax[1].plot(x, e['v_ext'] - e_ref['v_ext'], c=mpl.cm.get_cmap('tab10')(3))
    ax[1].plot(x, e['v_term'] - e_ref['v_term'], c=mpl.cm.get_cmap('tab10')(4))
    ax[1].plot(x, [0] * len(x), c='k', linestyle='--')
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig(f'{folder_name}/Energy_errors.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{folder_name}/Energy_errors.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # plot occupations
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
    ax.set_ylim(0, 2)
    # plt.savefig(f'{folder_name}/Densities.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{folder_name}/Densities.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # plot hxc potential
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    v_hxc_simple = np.array([j[-1] for j in v_hxc_progression_list])
    v_hxc_simple += (np.average(v_hxc_ref_progress, axis=1) - np.average(v_hxc_simple, axis=1)[np.newaxis]).T
    for i in range(6):
        plt.plot(x, v_hxc_simple[:, i], c=mpl.cm.get_cmap('tab10')(i), label=str(i))
        plt.plot(x, v_hxc_ref_progress[:, i], c=mpl.cm.get_cmap('tab10')(i), label=str(i) + '-ref', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel("v_hxc")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Trend of v_hxc. LPFET; values are corrected so averages match")
    # plt.savefig(f'{folder_name}/v_hxc_trend.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{folder_name}/v_hxc_trend.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.tight_layout()
    plt.xlabel(x_label)
    ax.set_ylabel('Error energy')
    for site in range(6):
        ax.plot(x, e_per_site[:, site]['tot'] - e_ref_per_site[:, site]['tot'], c=mpl.cm.get_cmap('tab10')(site),
                label=f'deviation on site {site}')
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.savefig(f'{folder_name}/Energy_errors_per_site.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{folder_name}/Energy_errors_per_site.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
