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

# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2', u_param=3,
#                blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2', u_param=5,
#                blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)
# generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2', u_param=8,
#                blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2)
generate_trend(8, 8, essentials.generate_chain1, '8-chain1_block-2_NNI-0.2', u_param=8,
               blocks=[[0, 1], [2, 3], [4, 5], [6, 7]], r_param=0.2, delta_x=0.01, max_value=0.2)


print('iterations: ', lpfet.ITERATION_NUM)

results.create_list.update_list()
