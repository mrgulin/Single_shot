import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime
lpfet.stream_handler.setLevel(20)

name = 'chain1'
mol1 = lpfet.Molecule(6, 6, name)
nodes_dict, edges_dict = essentials.generate_chain1(i=1, n_sites=6, U_param=1)
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
mol1.add_parameters(u, t, v_ext)

print(mol1.v_ext)
mol1.v_hxc = np.zeros(mol1.Ns)
# mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
mol1.find_solution_as_root()
# mol1.compare_densities_fci(False, True)

#
# name = 'chain1-block'
# mol2 = lpfet.MoleculeBlock(8, 8, name)
# nodes_dict, edges_dict = essentials.generate_random1(i=1, n_sites=8, U_param=1)
# t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
# mol2.add_parameters(u, t, v_ext)
# mol2.prepare_for_block([[0, 6], [1, 2], [3, 4], [5, 7]])
# mol2.find_solution_as_root()
#
# print(mol1.v_hxc, mol2.v_hxc)
