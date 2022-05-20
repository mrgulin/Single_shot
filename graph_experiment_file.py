from LPFET.calculate_batch import generate_trend
import LPFET.lpfet as lpfet
import LPFET.essentials as essentials
import results.create_list

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-332', i_param=1, blocks=[[0, 1, 2], [3, 4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-4', i_param=1, blocks=[[0, 1, 2, 3], [4, 5, 6, 7]])
generate_trend(8, 8, essentials.generate_chain1, '8-chain1', i_param=1, max_value=10, delta_x=0.2)

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2_v2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)

for n_electron in [2, 4, 6, 8, 10, 12, 14]:
    generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-block-2-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
                   r_param=0.2, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])

for n_electron in [2, 4, 6, 8, 10, 12, 14]:
    generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
                   r_param=0.2)



print('iterations: ', lpfet.ITERATION_NUM)

results.create_list.update_list()