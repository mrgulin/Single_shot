import lpfet

# Generate object
mol1 = lpfet.Molecule(6, 6, 'chain1')

# Generate arrays
pmv = 0.5
nodes_dict = dict()
edges_dict = dict()
eq_list = []
for j in range(6):
    nodes_dict[j] = {'v': -2.5 + j * pmv, 'U': 1}
    if j != 5:
        edges_dict[(j, j+1)] = 1
    eq_list.append([j])
t, v_ext, u = lpfet.generate_from_graph(nodes_dict, edges_dict)

# Add parameters
mol1.add_parameters(u, t, v_ext, eq_list)

# Main part of the program
mol1.self_consistent_loop(num_iter=20, tolerance=1E-6, oscillation_compensation=1)

# Comparison to FCI
y_ab, rest =mol1.compare_densities_FCI(pass_object=False)


