import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc_opt

import sys

sys.path.extend(['/mnt/c/Users/tinc9/Documents/CNRS-offline/', '../'])
import LPFET_homogeneous.lpfet_homogeneous


class MoleculeConstN(LPFET_homogeneous.lpfet_homogeneous.Molecule):
    def __init__(self, site_number, u, t, mu_ext, every_x=2, description=''):
        super().__init__(site_number, u, t, mu_ext, -1, description)
        self.every_x = every_x

    def set_n(self, new_n):
        self.Ne = new_n
        every_x = self.every_x
        i_floor = i_ceil = np.nan
        if every_x == 2:
            i_floor = int((self.Ne - 1e-10) // 2 * 2)
            i_ceil = i_floor + every_x
            if i_floor == -2:
                i_floor = 0
        elif every_x == 1:
            i_floor = int(np.floor(self.Ne))
            i_ceil = i_floor + every_x
        elif every_x == 4:
            ne = self.Ne
            if ne > self.Ns * 2 - 2:
                i_ceil = self.Ns * 2
                i_floor = i_ceil - 2
            elif ne < 2:
                i_floor = 0
                i_ceil = 2
            else:
                i_floor = int((self.Ne - 2 - 1e-10) // 4 * 4 + 2)
                i_ceil = i_floor + 4

        self.Ne_floor = i_floor
        self.Ne_ceil = i_ceil

        self.y_floor = self.precalculated_values[i_floor]['y']
        self.y_ceil = self.precalculated_values[i_ceil]['y']

        self.mu_ks_floor = self.precalculated_values[i_floor]['mu_ks']
        self.mu_ks_ceil = self.precalculated_values[i_ceil]['mu_ks']

        self.alpha = (new_n - i_floor) / (i_ceil - i_floor)

    def set_ext_pot(self):
        opt_v_imp_obj = sc_opt.minimize(optimization_function, np.array([0]),
                                        args=(self,),
                                        method='BFGS', options={'eps': 1e-3})
        error = opt_v_imp_obj['fun']
        ext_pot = opt_v_imp_obj['x'][0]
        print(f'{self.Ne:4.1f}: {ext_pot:6.2f} +- {error:8.4f}')
        return ext_pot


def optimization_function(ext_pot, mol_obj: MoleculeConstN):
    ext_pot2 = ext_pot[0]
    v_hxc_floor = ext_pot2 - mol_obj.mu_ks_floor
    v_hxc_ceil = ext_pot2 - mol_obj.mu_ks_ceil
    density_floor, energy_floor = LPFET_homogeneous.lpfet_homogeneous.casci_dimer(mol_obj.y_floor, mol_obj.h_ks,
                                                                                  mol_obj.u, -v_hxc_floor,
                                                                                  mol_obj.embedded_mol)
    density_ceil, energy_ceil = LPFET_homogeneous.lpfet_homogeneous.casci_dimer(mol_obj.y_ceil, mol_obj.h_ks, mol_obj.u,
                                                                                -v_hxc_ceil, mol_obj.embedded_mol)

    n_e_new = mol_obj.Ns * (mol_obj.alpha * density_ceil + (1 - mol_obj.alpha) * density_floor)

    print(f'\t{mol_obj.alpha:5.2f} {ext_pot2:8.5f} {density_ceil:6.2f}  {density_floor:6.2f} '
          f'{abs(n_e_new - mol_obj.Ne):6.3f}')
    return abs(n_e_new - mol_obj.Ne)


if __name__ == '__main__':
    obj = MoleculeConstN(6, 5, 1, -2, 2)
    result_list = []
    obj.precalculate_rdms()
    for electron_number in np.linspace(0.1, 11.9, 120 - 1):
        obj.set_n(electron_number)
        mu_ext = obj.set_ext_pot()
        result_list.append([electron_number, mu_ext])

    result_list = np.array(result_list)
    plt.plot(result_list[:, 0], result_list[:, 1])
    plt.plot(result_list[:, 0], result_list[:, 1])
    plt.ylim(-2, 5)
    plt.show()
