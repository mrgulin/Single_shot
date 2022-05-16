from LPFET.calculate_batch import generate_trend
import LPFET.lpfet as lpfet
import LPFET.essentials as essentials
import results.create_list

# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-1', i_param=1, blocks=[[0, 1], [2, 3], [4, 5]])
# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-2', i_param=1, blocks=[[0, 5], [1, 2], [3, 4]])
# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-no', i_param=1, blocks=None)

# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-1', i_param=2, blocks=[[0, 1], [2, 3], [4, 5]])
# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-2', i_param=2, blocks=[[0, 5], [1, 2], [3, 4]])
# time_l = generate_trend(6, 6, essentials.generate_ring4, 'ring4_block-no', i_param=2, blocks=None)

# time_l = generate_trend(6, 6, essentials.generate_chain1, 'chain1_block', i_param=1, blocks=[[0, 1], [2, 3], [4, 5]])

# time_l = generate_trend(8, 8, essentials.generate_chain1, '8-chain1', i_param=1)
# time_l = generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-1', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])
# time_l = generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2', i_param=1, blocks=[[0, 1, 2], [3, 4, 5], [6, 7]])


# generate_trend(8, 8, essentials.generate_random1, 'random1', i_param=1)
# generate_trend(8, 8, essentials.generate_random1, 'random1_block-1', i_param=1, blocks=[[0, 6], [1, 2], [3, 4], [5, 7]])
# generate_trend(8, 8, essentials.generate_random1, 'random1_block-2', i_param=1, blocks=[[0, 1], [2, 3, 4], [5, 6, 7]])
# generate_trend(8, 8, essentials.generate_random1, 'random1_block-2.2', i_param=1, blocks=[[0, 1, 2], [3, 4], [5, 6, 7]])

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-332', i_param=1, blocks=[[0, 1, 2], [3, 4, 5], [6, 7]])
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-4', i_param=1, blocks=[[0, 1, 2, 3], [4, 5, 6, 7]])

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2_v2', i_param=1, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)

for n_electron in [4, 6, 8, 10, 12]:
    generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-block-2-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
                   r_param=0.2, blocks=[[0, 1], [2, 3], [4, 5], [6, 7]])


print('iterations: ', lpfet.ITERATION_NUM)

results.create_list.update_list()
