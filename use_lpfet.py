import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime

name = 'chain1'
mol1 = lpfet.Molecule(6, 6, name)
mol_full = lpfet.class_Quant_NBody.QuantNBody(6, 6)
mol_full.build_operator_a_dagger_a()
first = False
nodes_dict, edges_dict = essentials.generate_ring4(i=1, n_sites=6, U_param=5)
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
mol1.add_parameters(u, t, v_ext, 0.2)
mol1.prepare_for_block([[0, 1], [2, 3], [4, 5]])
print(mol1.v_ext)
mol1.v_hxc = np.zeros(mol1.Ns)
mol1.v_hxc_progress = []
mol1.density_progress = []
start1 = datetime.now()
# mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
mol1.find_solution_as_root()
# mol1.find_solution_as_root()
mol1.compare_densities_fci(False, True)
