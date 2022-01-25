import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import single_shot

class CS_Householder(single_shot.Householder):  # Child class!!
    def add_starting_potential(self, mu_ext):
        self.mu_ext = mu_ext
        self.mu_KS = 0
        self.mu_Hxc = 0


    def self_consitent_loop(self):
        self.generate_huckel_hamiltonian(2)  # 2 for periodic condition

        self.calculate_eigenvectors()


        N_ele = 0
        for i in range(self.N): # Also size of matrix of 1RDM and ei_val
            if(self.ei_val[i] < self.mu_ext + 1E-10):
                N_ele += 2
        self.Ne = N_ele
        print(N_ele)
        self.generate_1rdm()
        self.n = self.Ne / self.N
        self.generate_householder_vector()
        self.calculate_variables()

    def loops(self):
        conv_test = 0
        for i in range(20):
            print(f"-----{i}------")
            self.Ne = self.vars['density'][0] * self.N
            self.n = self.Ne / self.N
            print("\tNelect before loop = ", self.Ne)
            conv_test = self.Ne
            # region DENSITY_MATRIX_BIS + SELFCONSISTENTLOOP_BIS

            # No need to calculate eigenvalues one more time if we are doing periodic conditions always!!
            print(self.Ne, ' eigenvalue number: ', int(round(self.Ne)//2), int(round(self.Ne)), 'associated mu: ',
                 self.ei_val[int(round(self.Ne)//2)])
            self.mu_KS = self.ei_val[int(round(self.Ne)//2)]
            self.mu_Hxc = self.mu_ext - self.mu_KS
            print(f'\tmu_KS {self.mu_KS}, mu_ext {self.mu_ext}, mu_Hxc = mu_KS-mu_ext {self.mu_Hxc}')

            print('\tCHECKING THAT LATTICE FILLING IS THE SAME AS IMPURITY :', self.n)
            print("new potential mu to build 1rdm ", self.mu_KS)
            N_ele = 0
            for j in range(self.N):  # Also size of matrix of 1RDM and ei_val
                if self.ei_val[j] < self.mu_KS + 1E-10:
                    N_ele += 2
            self.Ne = N_ele
            self.n = self.Ne / self.N

            self.generate_1rdm()
            self.mu_KS = self.ei_val[int(round(self.Ne) // 2)]
            self.generate_householder_vector()
            self.calculate_variables()
            print('\tafter interaction, impurity site occupation = ', self.vars['density'][0])
            print('\t', i, self.mu_ext, self.mu_KS, self.mu_Hxc, self.vars['d_occ'][0], self.e_site['main'], self.n)

            self.Ne = self.vars['density'][0] * self.N
            self.n = self.Ne / self.N
            print("\tNelect after loop = ", self.Ne)
            print('\tdifference:', conv_test - self.Ne)
            if abs(conv_test - self.Ne) < 1.0e-5:
                break
            # endregion


if __name__ == '__main__':
    obj = CS_Householder(400, 2, 4)
    obj.add_starting_potential(0)
    obj.self_consitent_loop()
    obj.loops()