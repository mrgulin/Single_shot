from LPFET.calculate_batch import generate_trend
import LPFET.lpfet as lpfet
import LPFET.essentials as essentials
import results.create_list
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
lpfet.stream_handler.setLevel(20)

a = datetime.now()

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-332', i_param=1, blocks=[[0, 1, 2], [3, 4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-4', i_param=1, blocks=[[0, 1, 2, 3], [4, 5, 6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1', i_param=1, max_value=10, delta_x=0.05)

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2_v2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)

# for n_electron in [2, 4, 6, 8, 10, 12, 14]:
#     generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-block-2-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
#                    r_param=0.2, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])
#
# for n_electron in [2, 4, 6, 8, 10, 12, 14]:
#     generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
#                    r_param=0.2)


# region random progression
times = []
for i in range(20):
    print(f'\n\ni={i}\n\n')
    a = datetime.now()
    mol1 = lpfet.MoleculeBlock(8, 8)
    nodes_dict, edges_dict = lpfet.essentials.generate_random1(1, 8, 5)
    t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
    mol1.add_parameters(u, t, v_ext)
    mol1.prepare_for_block([[0, 6], [1, 2], [3, 4], [5, 7]])
    mol1.find_solution_as_root(None)
    mol1.calculate_energy()
    t1 = datetime.now() - a
    print(f'time for calculations: {t1}, number of iterations: {lpfet.ITERATION_NUM}')
    b = datetime.now()
    mol1.compare_densities_fci()
    t2 = datetime.now() - b
    print(f'time for calculations: {t2}, number of iterations: {lpfet.ITERATION_NUM}')
    print(f'tot time: {datetime.now()-a}')
    times.append([t1, t2])
print(*times, sep='\n')
# n = np.array(mol1.density_progress)
# delta_n = np.array(mol1.optimize_progress_output)
# fig, ax = plt.subplots(1,1)
# for i in range(len(delta_n[0])):
#     ax.plot(np.abs(delta_n[:, i]), c=plt.get_cmap('viridis')(i/len(delta_n[0])))
# ax.set_yscale('log')
# ax.set_ylabel(r'''absolute density error $|e|$''')
# ax.set_xlabel(r'''Iteration number''')
# plt.savefig('thesis_graphs/iteration_error.svg', dpi=300, bbox_inches='tight')
# plt.savefig('thesis_graphs/iteration_error.eps', dpi=300, bbox_inches='tight')
# plt.savefig('thesis_graphs/iteration_error.png', dpi=300, bbox_inches='tight')
# plt.show()
# endregion



print('iterations: ', lpfet.ITERATION_NUM)

results.create_list.update_list()