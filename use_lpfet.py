import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime

name = 'chain1'
mol1 = lpfet.Molecule(6, 6, name)
mol_full = lpfet.class_Quant_NBody.QuantNBody(6, 6)
mol_full.build_operator_a_dagger_a()
first = False
nodes_dict, edges_dict, eq_list = essentials.generate_chain1(i=1, n_sites=6, U_param=1)
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
mol1.add_parameters(u, t, v_ext, eq_list, 0.5)
end_str = '\n'
print(mol1.v_ext)
mol1.v_hxc = np.zeros(mol1.Ns)
mol1.v_hxc_progress = []
mol1.density_progress = []
start1 = datetime.now()
# mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
mol1.find_solution_as_root()
end1 = datetime.now()
end_str += str(end1 - start1) + '\n'
mol1.compare_densities_fci(False, True)
print(end_str, lpfet.ITERATION_NUM)
