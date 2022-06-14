import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime
lpfet.stream_handler.setLevel(20)

# a = datetime.now()
# name = 'chain1'
# mol1 = lpfet.Molecule(8, 8, name)
# nodes_dict, edges_dict = essentials.generate_chain1(i=1, n_sites=8, U_param=1)
# t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
# mol1.add_parameters(u, t, v_ext)
#
# print(mol1.v_ext)
# mol1.v_hxc = np.zeros(mol1.Ns)
# # mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
# mol1.find_solution_as_root()
# # mol1.compare_densities_fci(False, True)
# print(datetime.now() - a, lpfet.ITERATION_NUM)
a = datetime.now()
mol1 = lpfet.Molecule(6, 6)
nodes_dict, edges_dict = lpfet.essentials.generate_chain1(1, 6, 5)
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
mol1.add_parameters(u, t, v_ext, 0.2, [[0,1], [2,3],[4,5]])
# mol1.prepare_for_block([[0, 6], [1, 2], [3, 4], [5, 7]])
mol1.find_solution_as_root(None)
mol1.calculate_energy()
t1 = datetime.now() - a
print(f'time for calculations: {t1}, number of iterations: {lpfet.ITERATION_NUM}')
b = datetime.now()
mol1.compare_densities_fci()
t2 = datetime.now() - b
print(f'time for calculations: {t2}, number of iterations: {lpfet.ITERATION_NUM}')
print(f'tot time: {datetime.now() - a}')

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
