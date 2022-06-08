import numpy as np
import psi4
import quantnbody as qnb
from scipy import linalg
import scipy.sparse.linalg
import math
from random import random
import LPFET.lpfet as lpfet

np.set_printoptions(precision=6) # For nice numpy matrix printing with numpy



# Generalproperties of the systems
N_MO   = 6   # Total number of spatial orbital
N_elec = 6   # Total number of electrons  
dim_H = math.comb( 2*N_MO, N_elec )  # Dimension of the many-electron Fockspace

# FERMI-HUBBARD MODEL 
t_,  Mu_  = np.zeros((N_MO,N_MO)),  np.zeros(N_MO)
U_ = np.zeros((N_MO,N_MO,N_MO,N_MO))
V_ = np.zeros((N_MO,N_MO,N_MO,N_MO))
for i in range(N_MO):
    U_[i,i,i,i]  =  .5 *(i+1) 
    Mu_[i]       = -2 -.1 *(i+1)
ratio = 0.3
for i in range(N_MO-1):
    # for j in range(i,N_MO):
        t_[i,i+1] = t_[i+1,i] = - 1
        V_[i, i, i + 1, i + 1] = V_[i+1, i+1, i, i] = ratio * (U_[i,i,i,i] + U_[i + 1, i + 1, i + 1, i + 1]) / 2
t_[0,N_MO-1] = t_[N_MO-1,0] = -1
V_[0, 0, N_MO - 1, N_MO - 1] = V_[N_MO - 1, N_MO - 1, 0 , 0] = ratio * (U_[0,0,0,0] + U_[N_MO - 1, N_MO - 1, N_MO - 1, N_MO - 1]) / 2
h_ = t_ + np.diag(Mu_)

mol1 = qnb.Hamiltonian(N_MO, N_elec)
mol1.build_operator_a_dagger_a()
mol1.build_hamiltonian_fermi_hubbard(h_, U_)
mol1.diagonalize_hamiltonian()
one_rdm = qnb.tools.build_1rdm_alpha(mol1.eig_vectors[:, 0], mol1.a_dagger_a)
P, v = lpfet.householder_transformation(one_rdm, 1)
h_ = P @ h_ @ P
V_ = np.einsum('ip, jq, kr, ls, ijkl -> pqrs', P, P, P, P, V_)
U_ = np.einsum('ip, jq, kr, ls, ijkl -> pqrs', P, P, P, P, U_)
mol1.build_hamiltonian_fermi_hubbard(h_, U_)
mol1.diagonalize_hamiltonian()
mol2 = qnb.Hamiltonian(N_MO, N_elec)
mol2.build_operator_a_dagger_a()
mol2.build_hamiltonian_fermi_hubbard(h_, U_, v_term=V_)
eig_energies, eig_vectors =  scipy.sparse.linalg.eigsh(mol1.H, k=10, which='SA' )

EXACT_GS_energy = eig_energies[0]
 

# %%

N_act      = 4
N_frozen   = 2
N_act_elec = N_elec - 2 * N_frozen 


frozen_indices = [0, 1]
active_indices = [2, 3, 4, 5]

mol_as = qnb.Hamiltonian(N_act, N_act_elec)
mol_as.build_operator_a_dagger_a()

Proj = np.zeros((dim_H,dim_H))
state_created = np.zeros(2*N_MO, dtype=np.int32)
for i in range(N_MO):
    if (i in frozen_indices):
        state_created[2*i]   = 1
        state_created[2*i+1] = 1 
        
for state in mol1.nbody_basis:
    if ( state_created @ state == 2 * N_frozen ):
            
        fstate_created = qnb.tools.my_state(state, mol1.nbody_basis)
        Proj += np.outer(fstate_created, fstate_created)
        print(state)
            

eig_energies, eig_vectors = scipy.sparse.linalg.eigsh(Proj @ mol1.H @ Proj, k=10, which='SA' )  #scipy.linalg.eigh( Proj @ H @ Proj )
eig_energies, eig_vectors = np.linalg.eigh(Proj @ mol1.H @ Proj)  #scipy.linalg.eigh( Proj @ H @ Proj )
eig_energies2, eig_vectors2 = scipy.sparse.linalg.eigsh(Proj @ mol2.H @ Proj, k=10, which='SA' )
EXACT_GS_energy = eig_energies2[0]
EXACT_GS_energy_no_v = eig_energies[0]

qnb.tools.visualize_wft(eig_vectors[:,0], mol1.nbody_basis)

print(eig_energies[:5])
 # %%


def get_active_space_integrals_fh_v_term(two_body_integrals,
                                         occupied_indices=None,
                                         active_indices=None):
    occupied_indices = [] if occupied_indices is None else occupied_indices
    if len(active_indices) < 1:
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        for j in occupied_indices:
            core_constant += 2 * 2 * two_body_integrals[i, i, j, j]
            core_constant += 2 * two_body_integrals[i, j, j, i]

    # Modified one electron integrals
    one_body_integrals_new = np.zeros(two_body_integrals.shape[:2])
    for u in active_indices:
        for i in occupied_indices:
            # one_body_integrals_new += two_body_integrals[i, u, u, i]
            for v in active_indices:
                one_body_integrals_new[u, v] += 2 * two_body_integrals[i, i, u, v]
                one_body_integrals_new[u, v] += 2 * two_body_integrals[u, v, i, i]
                one_body_integrals_new[u, v] -= two_body_integrals[i, v, u, i]

                # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[np.ix_(active_indices, active_indices)],
            two_body_integrals[np.ix_(active_indices, active_indices, active_indices, active_indices)])

core_energy_v, h_eff_v, v_term_act = get_active_space_integrals_fh_v_term(V_, occupied_indices=frozen_indices,
                                                                          active_indices=active_indices)

Core_energy, h_eff, U_act = qnb.tools.fh_get_active_space_integrals(h_,
                                                                       U_,
                                                                       frozen_indices=frozen_indices,
                                                                       active_indices=active_indices )


mol_as.build_hamiltonian_fermi_hubbard(h_eff + h_eff_v, U_act, v_term=v_term_act)
eig_energies_, eig_vectors_ = scipy.sparse.linalg.eigsh(mol_as.H, k=4, which='SA' )

AS_energy = Core_energy + eig_energies_[0] + core_energy_v
print(EXACT_GS_energy , AS_energy, -EXACT_GS_energy + AS_energy, EXACT_GS_energy_no_v)
print(111111)
