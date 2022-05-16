from LPFET.calculate_batch import generate_trend
import LPFET.lpfet as lpfet
import LPFET.essentials as essentials
import results.create_list
# generate_trend(6, 6, generate_chain2, 'chain2', u_param=5)
# generate_trend(6, 6, generate_chain3, 'chain3', u_param=5, delta_x=0.4, max_value=4)
# generate_trend(6, 6, essentials.generate_star1, 'star1', u_param=5, delta_x=0.4, max_value=4)
# generate_trend(6, 6, generate_complete1, 'complete1', u_param=5, delta_x=0.1, max_value=3)


# mol1 = lpfet.Molecule(6, 6, 'hehe')
# nodes_dict, edges_dict, eq_list = essentials.generate_chain1(2, 6, 10)
# t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)
# mol1.add_parameters(u, t, v_ext, eq_list)
# mol1.self_consistent_loop(50, 1e-4, oscillation_compensation=[5, 1])
# mol_full = lpfet.class_qnb.QuantNBody(6, 6)
# mol_full.build_operator_a_dagger_a()
# U = np.zeros((6, 6, 6, 6))
# for i in range(6):
#     U[i, i, i, i] = u[i]
# mol_full.build_hamiltonian_fermi_hubbard(t+np.diag(v_ext), U)
# mol_full.diagonalize_hamiltonian()
# tuple1 = mol_full.calculate_v_hxc(mol1.v_hxc)
# mol1.self_consistent_loop(num_iter=30, tolerance=1E-6, oscillation_compensation=[5, 1])
# time_l = generate_trend(6, 6, essentials.generate_chain1, 'chain1-root_r-0.5', i_param=1, r_param=0.5)
# time_l = generate_trend(6, 6, essentials.generate_chain1, 'chain1-root_r-0.2_test', i_param=1, r_param=0.2)
# time_l = generate_trend(6, 2, essentials.generate_star1, 'star1', i_param=1)
# time_l = generate_trend(6, 6, essentials.generate_star1, 'star1', i_param=1)
# time_l = generate_trend(6, 2, essentials.generate_complete1, 'complete1', i_param=3)
# time_l = generate_trend(6, 10, essentials.generate_complete1, 'complete1', i_param=3)




# generate_trend(8, 8, essentials.generate_chain1, '8-chain1', i_param=1, max_value=10, delta_x=0.2)
for n_electron in [2, 4, 6, 8, 10, 12, 14]:
    generate_trend(8, n_electron, essentials.generate_chain1, '8-chain1-NNI-0.2', i_param=1, max_value=10, delta_x=0.5,
                   r_param=0.2)


results.create_list.update_list()