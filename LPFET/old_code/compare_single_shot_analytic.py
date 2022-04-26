from single_shot import *

embedded_mol = LPFET.class_Quant_NBody.QuantNBody(2, 2)
embedded_mol.build_operator_a_dagger_a()
def CASCI_dimer(gamma, h, u, v_ext):
    print('start')
    P, v = Quant_NBody.Householder_transformation(gamma)
    y_a_tilde = P @ gamma @ P
    # h_tilde = np.einsum('ki, ij, jl -> kl', P, y_a_tilde, P)
    # print("!!!", np.allclose(P @ y_a_tilde @ P, np.einsum('ki, ij, jl -> kl', P, y_a_tilde, P)))
    h_tilde = P @ h @ P
    print_matrix(y_a_tilde)
    print('-')
    print_matrix(P)
    print('-')
    print_matrix(h)
    print('-')
    print_matrix(h_tilde)
    h_tilde_dimer = h_tilde[:2, :2]
    u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
    u_0_dimer[0, 0, 0, 0] += u
    h_tilde_dimer[0, 0] += v_ext
    print("--------\nh_tilde_dimer")
    print_matrix(h_tilde_dimer)

    # Subtract v_hcx
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()
    density_dimer = Quant_NBody.Build_One_RDM_alpha(embedded_mol.WFT_0, embedded_mol.a_dagger_a)
    # TODO: Change this with some object in class
    density = density_dimer[0, 0] * 2
    print("--------\ndensity")
    print_matrix(density_dimer)
    print(embedded_mol.ei_val)
    energy = embedded_mol.ei_val[0]
    print('stop')
    return density, energy

if __name__ == "__main__":
    """obj = Householder(10, 4, 8)
    obj.calculate_one()
    print(obj.results_string)"""
    # print(obj.procedure_log)
    # calculate_many_conditions(100, 8)
    # calculate_many_conditions(10, 8)
    obj = Householder(10, 10, 4, debug=True)
    # Huckel hamiltonian generation: self.h in our case
    obj.generate_huckel_hamiltonian()

    # Generating eigenvalues and eigenvectors
    obj.calculate_eigenvectors()

    obj.generate_1rdm()

    # Householder vector generation
    obj.generate_householder_vector()

    density, energy = CASCI_dimer(obj.gamma, obj.h, obj.U, 0)

    obj.calculate_variables()
    print(obj.vars)
    print(density)
