import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime
lpfet.stream_handler.setLevel(2)

name = 'chain1'
mol1 = lpfet.Molecule(8, 8, name)
first = False
nodes_dict, edges_dict = essentials.generate_random1(i=1, n_sites=8, U_param=1)
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
mol1.add_parameters(u, t, v_ext)
mol1.prepare_for_block([[0, 1], [2, 3, 4], [5, 6, 7]])
print(mol1.v_ext)
mol1.v_hxc = np.zeros(mol1.Ns)
mol1.v_hxc_progress = []
mol1.density_progress = []
start1 = datetime.now()
# mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
mol1.find_solution_as_root()
# mol1.find_solution_as_root()
mol1.compare_densities_fci(False, True)
