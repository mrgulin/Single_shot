import matplotlib.pyplot as plt
import LPFET.lpfet as lpfet
import numpy as np
import sys
import LPFET.essentials as essentials
from datetime import datetime

lpfet.stream_handler.setLevel(20)
from tqdm import tqdm
import quantnbody as qnb
import math
from ab_initio_reference.FCI import calculate_reference
import quantnbody_class_new as qnb_class

# list_theta = np.linspace(num=30, start=20. * np.pi / 180., stop=160. * np.pi / 180., endpoint=True)
list_theta = np.linspace(0.35, 3, 30)
# list_theta = [1.2, 1.3]
E_HF = []
E_FCI = []
E_lpfet = []
E_FCI_QNB = []

# N_MO = 8
# N_elec = 8
# blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]  # [[i] for i in range(8)]  # [[0, 1], [2, 3], [4, 5], [6, 7]]
# eq_sites = [[0, 7],[1, 6], [2, 5], [3, 4]]  # [[0, 7], [1, 6], [2, 5], [3, 4]]  # [[0, 6], [1, 7], [2, 4], [3, 5]]  #
# lpfet.ACTIVE_SPACE_CALCULATION = True
# name_system = 'H8_chain_cluster2222_non-int-bath_AS'
N_MO = 6
N_elec = 6
blocks = [[0, 1], [2, 3], [4, 5]]  # [[i] for i in range(8)]  # [[0, 1], [2, 3], [4, 5], [6, 7]]
eq_sites = [[0, 1, 2, 3, 4, 5]]  # [[0, 7], [1, 6], [2, 5], [3, 4]]  # [[0, 6], [1, 7], [2, 4], [3, 5]]  #
lpfet.ACTIVE_SPACE_CALCULATION = True
name_system = 'H6_ring_cluster222_non-int-bath_AS'
name = fr'C:\Users\tinc9\Documents\CNRS-offline\internship_project\results\{name_system}/'
MAX_ROOT = 15
elem = 0
dim_parameter = len(list_theta)
calculate_qnb = False
qnb_obj = qnb_class.HamiltonianV2(N_MO, N_elec)
qnb_obj.build_operator_a_dagger_a()
x = []
my_mol = lpfet.MoleculeChemistry(N_MO, N_elec)
lpfet.stream_handler.setLevel(5)
old_approximation = None
progress = {'y_fci': [], 'y_lpfet': [], 'E_1rdm': [], 'E_2rdm': [], 'E_1rdm_lpfet': [], 'E_2rdm_lpfet': []}
for theta in tqdm(list_theta, file=sys.stdout):
    r = 1

    # XYZ_geometry = """ H   {0}   {1}  0.
    #                    H   {0}  -{1}  0.
    #                    H  -{0}   {1}  0.
    #                    H  -{0}  -{1}  0.  """.format(r * np.cos(theta / 2.), r * np.sin(theta / 2.))

    # XYZ_geometry = qnb.tools.generate_h_ring_geometry(6, theta) + "\n symmetry c1"
    # angle = 104.45 * np.pi / 180
    # XYZ_geometry = """ O   0    0  0.
    #                    H   {0}  0  0.
    #                    H   {1}  {2} 0.""".format(theta, np.cos(angle) * theta, np.sin(angle) * theta)

    # XYZ_geometry = qnb.tools.generate_h_chain_geometry(6, theta)

    # XYZ_geometry = qnb.tools.generate_h_ring_geometry(6, theta)

    XYZ_geometry = qnb.tools.generate_h_ring_geometry(6, theta)

    # range1 = np.arange(8)
    # XYZ_geometry = '\n'.join([f"H    0.    {i}   0." for i in theta * (range1 // 2) + 0.8 * ((range1 + 1) // 2)])

    try:
        ret_dict = calculate_reference(XYZ_geometry, basis='STO-3G', use_hf_orbitals=False)
    except:
        print("Couldn't calculate reference FCI")
        continue
    x.append(theta)
    # basis='STO-3G' -->  minimal basis sets
    # besis='6-31G' --> more than minimal basis set

    h, g, nuc_rep, wf_full, nbody_basis_psi4, eig_val, Ham, C = (ret_dict['h'], ret_dict['g'], ret_dict['nuc_rep'],
                                                                 ret_dict['wave_function'], ret_dict['det_list'],
                                                                 ret_dict['eig_val'], ret_dict['H'], ret_dict['C'])
    E_HF.append(ret_dict['HF'])
    E_FCI.append(ret_dict['FCI'])
    if calculate_qnb:
        print("generating Hamiltonian")
        qnb_obj.build_hamiltonian_quantum_chemistry(h, g)
        print("Diagonalizing Hamiltonian")
        qnb_obj.diagonalize_hamiltonian(False)
        print("calculation")
        one_rdm = qnb_obj.calculate_1rdm_spin_free(0)
        progress['E_1rdm'].append(np.sum(one_rdm * h))
        progress['E_2rdm'].append(qnb_obj.eig_values[0] - progress['E_1rdm'][-1])
        progress['y_fci'].append(one_rdm)
        print(qnb_obj.eig_values)
        E_FCI_QNB.append(qnb_obj.eig_values[0] + nuc_rep)
    else:
        progress['E_1rdm'].append(0)
        progress['E_2rdm'].append(0)
        progress['y_fci'].append(np.zeros((N_MO, N_MO)))
        E_FCI_QNB.append(0)

    # nbody_basis_v2 = []
    # for a in nbody_basis_psi4:
    #     nbody_basis_v2.append(np.zeros(2 * N_MO))
    #     for id1 in a.obtBits2ObtIndexList(a.alphaObtBits):
    #         nbody_basis_v2[-1][2 * id1] += 1
    #     for id1 in a.obtBits2ObtIndexList(a.betaObtBits):
    #         nbody_basis_v2[-1][2 * id1 + 1] += 1
    # ta = []
    # for det in nbody_basis_v2:
    #     ta.append(np.argwhere(np.all(det == qnb_obj.nbody_basis, axis=1))[0][0])
    # wf_0 = np.zeros(len(qnb_obj.nbody_basis))
    # wf_0[ta] = wf_full[:, 0]
    # print(qnb.tools.build_1rdm_spin_free(wf_0, qnb_obj.a_dagger_a))

    my_mol.clear_object()
    my_mol.v_ext = np.zeros(N_MO)
    # my_mol.add_parameters(g, h, [[i] for i in range(N_MO)], 0)
    my_mol.add_parameters(g, h, eq_sites, blocks=blocks)

    # my_mol.prepare_for_block([[i] for i in range(N_MO)])
    # my_mol.prepare_for_block([[2 * i, 2 * i + 1] for i in range(N_MO//2)])
    my_mol.ab_initio = True
    old_approximation = my_mol.find_solution_as_root(old_approximation)
    E_lpfet.append(my_mol.calculate_energy() + nuc_rep)
    print(E_lpfet[-1], ret_dict['FCI'])
    progress['E_1rdm_lpfet'].append(my_mol.energy_contributions[1])
    progress['E_2rdm_lpfet'].append(my_mol.energy_contributions[3])
    progress['y_lpfet'].append(my_mol.y_a)
    first1 = True
    # fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # for jj in range(N_MO):
    #     ax.plot(np.array(my_mol.density_progress)[:, jj], label='KS density', c='b')
    #     first1 = False
    # # plt.plot(np.sum(my_mol.density_progress, axis=1), label='total number e KS', c='k')
    # inputs = np.array(my_mol.optimize_progress_input)
    # ax2 = ax.twinx()
    # ax2.plot(inputs[:, 0], label='impurity potential')
    # for jj in range(1, len(inputs[0])):
    #     ax.plot(inputs[:, jj], c='g', label='$v^Hxc$')
    # cluster_densities = np.array(my_mol.density_progress)[:, [i[0] for i in my_mol.equiv_atoms_in_block]]
    # for jj in range(len(cluster_densities[0])):
    #     ax.plot(cluster_densities[:, jj], c='r')
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig(f'{name}progression_{theta:.3f}.png', dpi=150, bbox_inches='tight')
    # plt.show()
    try:
        pass
    except BaseException as e:
        print(e)
        E_lpfet.append(np.nan)

    elem += 1

E_lpfet = np.array(E_lpfet)
E_FCI = np.array(E_FCI)
E_HF = np.array(E_HF)
E_FCI_QNB = np.array(E_FCI_QNB)
progress['y_lpfet'] = np.array(progress['y_lpfet'])
progress['y_fci'] = np.array(progress['y_fci'])

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
plt.xlabel('$r$')
plt.ylabel("$E$")
ax.plot(x, E_lpfet, label='lpfet', linestyle='dashed')
ax.plot(x, E_FCI, label='ref', c='k', linewidth=0.75)
ax.plot(x, E_HF, label='HF', linestyle='dashed')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(f"Disociation curve for {name_system}")
ax.set_ylim(top=0)
plt.savefig(f'{name}disociation_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{name}disociation_curve.svg', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.set_xlabel('$r$')
ax.set_ylabel(r"""$\Delta E$""")
ax.plot(x, np.array(progress['E_1rdm_lpfet']) - np.array(progress['E_1rdm']),
        label='one electron interaction error')
ax.plot(x, np.array(progress['E_2rdm_lpfet']) - np.array(progress['E_2rdm']),
        label='two electron interaction error')
plt.axhline(0, linestyle='--', color='k', linewidth=0.8)
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(f"Error in contributions for {name_system}")
plt.savefig(f'{name}contribution_errors.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{name}contribution_errors.svg', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.set_xlabel('$r$')
ax.set_ylabel(r"""$\Delta E$""")
ax.plot(x, E_lpfet - E_FCI, label='lpfet')
ax.plot(x, E_HF - E_FCI, label='HF')
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(f"Errors in energy for {name_system}")
plt.savefig(f'{name}relative_errors.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{name}relative_errors.svg', dpi=150, bbox_inches='tight')
plt.show()

np.savetxt(name + 'data.dat',
           np.array([x, E_FCI, E_FCI_QNB, E_HF, E_lpfet, progress['E_1rdm'], progress['E_1rdm_lpfet'],
                     progress['E_2rdm'], progress['E_2rdm_lpfet'],
                     *[[j[i, i] for j in progress['y_fci']] for i in range(len(progress['y_fci'][0]))],
                     *[[j[i, i] for j in progress['y_lpfet']] for i in
                       range(len(progress['y_lpfet'][0]))]
                     ]).T, header='list_theta, E_FCI, E_FCI_QNB, E_HF, E_lpfet, E_1rdm, E_1rdm_lpfet, E_2rdm'
                                  ', E_2rdm_lpfet, N*n_fci, N*N_lpfet')
