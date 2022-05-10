import sys
sys.path.append('/mnt/c/Users/tinc9/Documents/CNRS-offline/QuantNBody/')

import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime
lpfet.stream_handler.setLevel(20)
from tqdm import tqdm
import quantnbody as qnb
import math
from ab_initio_reference.FCI import calculate_reference
import quantnbody_class_new as qnb_class

# list_theta = np.linspace(num=30, start=20. * np.pi / 180., stop=160. * np.pi / 180., endpoint=True)
list_theta = np.linspace(0.25, 5, 30)
list_r = np.linspace(0.25, 3, 25)
# list_theta = [1.2, 1.3]
E_HF = []
E_FCI = []
E_FCI_me = []
E_FCI_QNB = []

N_MO = 6
N_elec = 6
MAX_ROOT = 15
elem = 0
dim_parameter = len(list_theta)

qnb_obj = qnb_class.HamiltonianV2(N_MO, N_elec)
qnb_obj.build_operator_a_dagger_a()

my_mol = lpfet.MoleculeBlockChemistry(N_MO, N_elec)
lpfet.stream_handler.setLevel(5)
progress = {'y_fci': [], 'y_lpfet': [], 'E_1rdm': [], 'E_2rdm': [], 'E_1rdm_lpfet': [], 'E_2rdm_lpfet': []}
for theta in tqdm(list_theta, file=sys.stdout):
    r = 1

    # XYZ_geometry = """ H   {0}   {1}  0.
    #                    H   {0}  -{1}  0.
    #                    H  -{0}   {1}  0.
    #                    H  -{0}  -{1}  0.  """.format(r * np.cos(theta / 2.), r * np.sin(theta / 2.))

    XYZ_geometry = qnb.tools.generate_h_ring_geometry(6, theta) + "\n symmetry c1"

    ret_dict = calculate_reference(XYZ_geometry, basis='STO-3G') # basis='STO-3G'

    h, g, nuc_rep, wf_full, nbody_basis_psi4, eig_val, Ham, C = (ret_dict['h'], ret_dict['g'], ret_dict['nuc_rep'],
                                                                      ret_dict['wave_function'], ret_dict['det_list'],
                                                                 ret_dict['eig_val'], ret_dict['H'], ret_dict['C'])
    E_HF.append(ret_dict['HF'])
    E_FCI.append(ret_dict['FCI'])

    qnb_obj.build_hamiltonian_quantum_chemistry(h, g)
    qnb_obj.diagonalize_hamiltonian(False)
    one_rdm = qnb_obj.calculate_1rdm_spin_free(0)
    progress['E_1rdm'].append(np.sum(one_rdm * h))
    progress['E_2rdm'].append(qnb_obj.eig_values[0] - progress['E_1rdm'][-1])
    progress['y_fci'].append(one_rdm)
    print(qnb_obj.eig_values)
    E_FCI_QNB.append(qnb_obj.eig_values[0] + nuc_rep)

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

    my_mol.clear_object('')
    my_mol.v_ext = np.zeros(N_MO)
    # my_mol.add_parameters(g, h, [[i] for i in range(N_MO)], 0)
    my_mol.add_parameters(g, h, [list(range(N_MO))])

    # my_mol.prepare_for_block([[i] for i in range(N_MO)])
    # my_mol.prepare_for_block([[2 * i, 2 * i + 1] for i in range(N_MO//2)])
    my_mol.prepare_for_block([[0, 1, 2], [3, 4, 5]])
    my_mol.ab_initio = True
    my_mol.find_solution_as_root(None)
    E_FCI_me.append(my_mol.calculate_energy() + nuc_rep)
    progress['E_1rdm_lpfet'].append(my_mol.energy_contributions[1])
    progress['E_2rdm_lpfet'].append(my_mol.energy_contributions[3])
    progress['y_lpfet'].append(my_mol.y_a)
    try:
        pass
    except BaseException as e:
        print(e)
        E_FCI_me.append(np.nan)

    elem += 1

E_FCI_me = np.array(E_FCI_me)
import matplotlib.pyplot as plt
plt.plot(list_theta, E_FCI_me, label='lpfet',  linestyle='dashed')
plt.plot(list_theta, E_FCI, label='ref', c='k')
plt.plot(list_theta, E_FCI_QNB, label='qnb_fci',  linestyle='dashed')
plt.plot(list_theta, E_HF, label='HF', linestyle='dashed')
plt.legend()
plt.ylim(top=2)
plt.title('minimal basis set H-ring 6 sites stretching n_impurity=3')
plt.show()
plt.plot(list_theta, np.array(progress['E_1rdm_lpfet']) - np.array(progress['E_1rdm']),
         label='one electron interaction error')
plt.plot(list_theta, np.array(progress['E_2rdm_lpfet']) - np.array(progress['E_2rdm']),
         label='two electron interaction error')
plt.axhline(0, linestyle='--', color='k', linewidth=0.8)
plt.legend()
plt.title('minimal basis set H-ring 6 sites stretching n_impurity=3')
plt.show()
