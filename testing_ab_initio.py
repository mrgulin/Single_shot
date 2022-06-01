import matplotlib.pyplot as plt
import LPFET.lpfet as lpfet
import numpy as np
import LPFET.essentials as essentials
from datetime import datetime

lpfet.stream_handler.setLevel(20)
from tqdm import tqdm
import quantnbody as qnb
import math
from ab_initio_reference.FCI import calculate_reference
import quantnbody_class_new as qnb_class


XYZ_geometry = qnb.tools.generate_h_chain_geometry(4, 0.5)

ret_dict = calculate_reference(XYZ_geometry, basis='STO-3G', use_hf_orbitals=False)
h, g, nuc_rep, wf_full, nbody_basis_psi4, eig_val, Ham, C = (ret_dict['h'], ret_dict['g'], ret_dict['nuc_rep'],
                                                             ret_dict['wave_function'], ret_dict['det_list'],
                                                             ret_dict['eig_val'], ret_dict['H'], ret_dict['C'])
my_mol = lpfet.MoleculeBlockChemistry(4, 4)
my_mol.add_parameters(g, h, [[0, 3], [1, 2]])
my_mol.prepare_for_block([[i] for i in range(4)])
my_mol.ab_initio = True
my_mol.find_solution_as_root()