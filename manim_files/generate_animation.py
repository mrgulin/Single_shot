import numpy as np
import Quant_NBody
import LPFET.lpfet as lpfet
import essentials
from manim import *

dimer_opt_list = []

def cost_function_CASCI(mu_imp, embedded_mol, h_tilde_dimer, u_0_dimer, desired_density):
    mu_imp = mu_imp[0]
    mu_imp_array = np.array([[mu_imp, 0], [0, 0]])
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer - mu_imp_array, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()
    density_dimer = embedded_mol.calculate_1rdm(index=0)
    dimer_opt_list.append([mu_imp, density_dimer[0,0], desired_density])
    return (density_dimer[0, 0] - desired_density) ** 2

def decimal_matrix(matrix):
    return DecimalMatrix(matrix, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)

class MoleculeData(lpfet.Molecule):
    def __init__(self, Ns, Ne, sites, connections, equiv_atom_groups):
        super().__init__(Ns, Ne, "Data for molecule")
        self.param_dict = sites
        self.t_dict = connections
        self.equivalent_sites = equiv_atom_groups
        t, v, u = lpfet.generate_from_graph(sites, connections)
        super().add_parameters(u, t, v, equiv_atom_groups)
        super().calculate_ks()

        # hh transformation
        self.v = np.array([])
        self.P = np.array([])

        self.h_tilde = np.array([])
        self.h_tilde_dimer = np.array([])
        self.dimer_opt_list = np.array([])
        self.mu_imp = -10

    def get_access_to_variables_casci(self, site_id):
        global dimer_opt_list

        # Householder transforms impurity on index 0 so we have to make sure that impurity is on index 0:
        y_a_correct_imp = lpfet.change_indices(self.y_a, site_id)
        t_correct_imp = lpfet.change_indices(self.t, site_id)
        v_s_correct_imp = lpfet.change_indices(self.v_s, site_id)

        self.P, self.v = Quant_NBody.householder_transformation(y_a_correct_imp)
        self.h_tilde = self.P @ (t_correct_imp + np.diag(v_s_correct_imp)) @ self.P

        h_tilde_dimer = self.h_tilde[:2, :2]
        u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
        u_0_dimer[0, 0, 0, 0] += self.u[site_id]
        # h_tilde_dimer[0,0] += self.v_ext[site_id]
        mu_imp = self.v_hxc[[site_id]]  # Double parenthesis so I keep array, in future this will be list of
        # indices for block householder

        self.h_tilde_dimer = h_tilde_dimer
        opt_v_imp_obj = lpfet.sc_opt.minimize(cost_function_CASCI, mu_imp,
                                        args=(self.embedded_mol, h_tilde_dimer, u_0_dimer, self.n_ks[site_id]),
                                        method='BFGS', options={'eps': 1e-5})
        # This minimize cost function (difference between KS occupations and CASCI occupations squared)
        error = opt_v_imp_obj['fun']
        mu_imp = opt_v_imp_obj['x'][0]

        self.dimer_opt_list = np.array(dimer_opt_list)
        dimer_opt_list = []
        self.mu_imp = mu_imp

        # self.update_v_hxc(site_group, mu_imp, oscillation_compensation)
        #
        # on_site_repulsion_i = self.embedded_mol.calculate_2rdm_fh(index=0)[0, 0, 0, 0] * u_0_dimer[0, 0, 0, 0]
        # for every_site_id in self.equiv_atom_groups[site_group]:
        #     self.kinetic_contributions[every_site_id] = 2 * h_tilde[1, 0] * self.embedded_mol.one_rdm[1, 0]
        #     self.onsite_repulsion[every_site_id] = on_site_repulsion_i
        #     self.imp_potential[every_site_id] = mu_imp


v_ext_amplitude = 1
param_dict = dict()
for i in range(4):
    param_dict[i] = {'v': v_ext_amplitude * (-1) ** i, 'U': 3}



# Ns = 4
# v_ext = np.array([1, -1, 1, -1])
# v_hxc = np.array([0, 0, 0, 0])
# t = np.array([[0, -1, 0, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, -1, 0]])
# eig_vec = np.random.random((Ns, Ns))
# eig_val = np.random.random(Ns)
# gamma = np.zeros((Ns, Ns))
#
# Ne = 4

def transform_into(old_object, new_object):
    return Transform(old_object, new_object.move_to(old_object.get_center()))


class MatrixExamples(Scene):
    def construct(self):
        m0 = Matrix([[2, "\pi"], [-1, 1]])
        m1 = Matrix([[2, 0, 4], [-1, 1, 5]],
                    v_buff=1.3,
                    h_buff=0.8,
                    bracket_h_buff=SMALL_BUFF,
                    bracket_v_buff=SMALL_BUFF,
                    left_bracket="\{",
                    right_bracket="\}")
        m1.add(SurroundingRectangle(m1.get_columns()[1]))
        m2 = Matrix([[2, 1], [-1, 3]],
                    element_alignment_corner=UL,
                    left_bracket="(",
                    right_bracket=")")
        m3 = Matrix([[2, 1], [-1, 3]],
                    left_bracket="\\langle",
                    right_bracket="\\rangle")
        m4 = Matrix([[2, 1], [-1, 3]],
                    ).set_column_colors(RED, GREEN)
        m5 = Matrix([[2, 1], [-1, 3]],
                    ).set_row_colors(RED, GREEN)
        g = Group(
            m0, m1, m2, m3, m4, m5
        ).arrange_in_grid(buff=2)
        self.add(g)
        tex = Tex(r"\LaTeX", font_size=144)
        self.add(tex)


class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*(FadeOut(obj, shift=DOWN) for obj in basel)),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid",
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ],
                ),
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()


class OpeningManim2(Scene):
    def construct(self):
        title = Tex(r"This is some KS Hamiltonian")
        ks_hamiltonian = MathTex(r"\hat H ^{KS} = \hat T + \hat V^{s}")
        ks_hamiltonian2 = MathTex(r"\hat H ^{KS} = \hat T + \hat V^{ext} + \hat V^{Hxc}")
        print(ks_hamiltonian)
        scene1 = VGroup(title, ks_hamiltonian).arrange(DOWN)
        scene2 = VGroup(title, ks_hamiltonian2).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(ks_hamiltonian)
        )
        self.wait()

        self.play(
            ReplacementTransform(scene1, scene2)
        )
        self.wait()

        self.play(
            FadeOut(scene2)
        )
        self.wait()


class OpeningManim3(Scene):
    def construct(self):
        text_v_ext = MathTex(r"\vec{v}^s")
        text_v_hxc = MathTex(r"\vec{v}^{Hxc}")

        m_v_ext_vec = Matrix(mol1.v_ext.reshape(-1, 1))
        m_v_hxc_vec = Matrix(mol1.v_hxc.reshape(-1, 1))

        scene3 = VGroup(VGroup(text_v_ext, m_v_ext_vec).arrange(DOWN, buff=1),
                        VGroup(text_v_hxc, m_v_hxc_vec).arrange(DOWN, buff=1)).arrange(RIGHT, buff=8)
        self.play(FadeIn(scene3))
        self.wait()

        m_v_ext = Matrix(np.diag(mol1.v_ext))

        m_v_ext.move_to(scene3[0][1])

        m_v_hxc = Matrix(np.diag(mol1.v_hxc))
        m_v_hxc.move_to(scene3[1][1])

        self.play(Transform(scene3[0][1], m_v_ext), Transform(scene3[1][1], m_v_hxc))
        self.wait()

        m_t = Matrix(t)

        plus1 = MathTex(r"+")
        plus2 = MathTex(r"+")
        ks_hamiltonian3 = MathTex(r"H ^{KS} = ")

        scene4 = VGroup(ks_hamiltonian3, m_t, plus1, m_v_ext, plus2, m_v_hxc).arrange(RIGHT, buff=0.2).scale(0.65)
        text_v_ext2 = scene3[0][0].move_to(scene4[3].get_center() + UP * 2)
        text_v_hxc2 = scene3[1][0].move_to(scene4[5].get_center() + UP * 2)
        text_t = MathTex(r"t").move_to(scene4[1].get_center() + UP * 2)
        self.play(
            FadeIn(text_t),
            ReplacementTransform(scene3[0][0], text_v_ext2),
            ReplacementTransform(scene3[1][0], text_v_hxc2),
            ReplacementTransform(scene3[0][1], scene4[3]),
            ReplacementTransform(scene3[1][1], scene4[5]),
            FadeIn(scene4[0:3]), FadeIn(scene4[4]))
        self.wait()

        text_v_s = MathTex(r"v^s").move_to(scene4[3].get_center() + UP * 2).scale(0.65)
        m_v_s = decimal_matrix(np.diag(mol1.v_ext) + np.diag(mol1.v_hxc)).move_to(scene4[3]).scale(0.65)
        self.play(
            ReplacementTransform(text_v_ext, text_v_s),
            ReplacementTransform(m_v_hxc, m_v_s),
            FadeOut(m_v_ext),
            FadeOut(plus2),
            FadeOut(text_v_hxc)
        )

        text_h_ks_top = MathTex(r"H^{KS}").move_to(scene4[1].get_center() + UP * 2).scale(0.65)
        m_h_ks = decimal_matrix(np.diag(mol1.v_ext) + np.diag(mol1.v_hxc) + t).move_to(scene4[1]).scale(0.65)
        self.play(
            ReplacementTransform(text_v_s, text_h_ks_top),
            ReplacementTransform(m_v_s, m_h_ks),
            FadeOut(m_t),
            FadeOut(text_t),
            FadeOut(plus1),

        )
        self.wait()
        self.play(FadeOut(text_h_ks_top, m_h_ks, ks_hamiltonian3))
        self.wait()


class Generate_ring(Scene):
    def construct(self):
        mol1 = MoleculeData(4, 4, param_dict, {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, [[0, 2], [1, 3]])
        self.add(NumberPlane())
        theta = np.pi / 4
        r = 1.5
        points = []
        lines = []
        sites_param = []
        edge_param = []
        for index1, i in enumerate(np.linspace(theta, np.pi * 2 + theta, mol1.Ns, False)):
            points.append(Circle().move_to([r * np.cos(i), r * np.sin(i), 0]).set_color(WHITE).scale(0.3))
            sites_param.append(
                Text(f'U={param_dict[index1]["U"]}, v_ext={param_dict[index1]["v"]}', font_size=23).move_to(
                    [1.5 * r * np.cos(i), 1.5 * r * np.sin(i), 0]))
            if index1 != mol1.Ns - 1:
                edge_param.append(
                    Text(f't={mol1.t_dict[(index1, index1 + 1)]}', font_size=23).move_to(
                        [1.5 * r * np.cos(i + np.pi / mol1.Ns), 1.5 * r * np.sin(i + np.pi / mol1.Ns), 0]))

            if index1 % 2 == 0:
                points[-1].set_fill(WHITE, opacity=1)
            else:
                points[-1].set_fill(BLACK, opacity=1)
        for i in range(len(points) - 1):
            lines.append(Line(points[i].get_center(), points[i + 1].get_center()).set_color(WHITE))
        edge_param.append(
            Text(f't={mol1.t_dict[(len(points) - 1, 0)]}', font_size=23).move_to(
                [1.25 * r * np.cos(theta - np.pi / mol1.Ns), 1.25 * r * np.sin(theta - np.pi / mol1.Ns), 0]))
        lines.append(Line(points[0].get_center(), points[-1].get_center()).set_color(WHITE))
        molecule = Group(*lines, *points)
        sites_param = Group(*sites_param)
        edge_param = Group(*edge_param)
        text_electrons = Text(f'{mol1.Ne}e', font_size=50)

        title1 = Text(r"Generation of the system", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(FadeIn(title1))
        self.play(GrowFromCenter(molecule))
        self.wait()
        self.play(GrowFromCenter(sites_param))
        self.wait()
        self.play(FadeOut(sites_param))
        self.wait()
        self.play(GrowFromCenter(edge_param))
        self.wait()
        self.play(FadeOut(edge_param))
        self.wait()
        self.play(molecule.animate.scale(0.6).move_to([6, 3, 0]))
        self.wait()
        self.play(GrowFromCenter(text_electrons))
        self.play(text_electrons.animate.move_to([6, 3, 0]))
        self.wait()

        v_hxc_horizontal = np.array([[0.0, 0.00, 0.01, 0]])
        m_v_ext_vec = decimal_matrix(v_hxc_horizontal).scale(0.75)

        title = Tex(r"Starting Hxc approximation").move_to([0, 1, 0])
        vec1 = MathTex(r"\vec v^{Hxc} = ")
        v_hxc_horizontal = Group(vec1, m_v_ext_vec).arrange(RIGHT, buff=0.5).move_to([0, -1, 0])
        self.play(FadeIn(title, v_hxc_horizontal))
        self.wait()
        self.play(FadeOut(title, title1), v_hxc_horizontal.animate.move_to([2.9, 3, 0]).scale(0.75))
        y0 = 2
        x0 = 0.5
        separating_line1 = Line([7, y0, 0], [-7, y0, 0])
        separating_line2 = Line([x0, y0, 0], [x0, 4, 0])
        self.play(Create(separating_line1))
        self.play(Create(separating_line2))
        self.wait()
        title1 = Text(r"Kohn-Sham system", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(Write(title1))

        title = Tex(r"KS Hamiltonian").move_to([0, 0.5, 0])
        ks_hamiltonian = MathTex(r"\hat H ^{KS} = \hat T + \hat V^{s}").move_to([0, -1.5, 0])
        print(ks_hamiltonian)
        self.play(
            Write(title),
            FadeIn(ks_hamiltonian)
        )
        self.wait()
        self.play(
            Transform(ks_hamiltonian,
                      MathTex(r"\hat H ^{KS} = \hat T + ", r"\hat V^{ext}", "+", r"\hat V^{Hxc}").move_to([0, -1.5, 0]))
        )
        self.wait()
        self.play(
            FadeOut(title, ks_hamiltonian[0], ks_hamiltonian[2]),
            ks_hamiltonian[1].animate.move_to([-4, 1, 0]),
            ks_hamiltonian[3].animate.move_to([4, 1, 0])
        )
        self.wait()

        # Matrix calculation of KS Hamiltonian
        text_v_ext = ks_hamiltonian[1]
        text_v_hxc = ks_hamiltonian[3]

        self.play(
            transform_into(text_v_ext, MathTex(r"\vec{v}^{ext}")),
            transform_into(text_v_hxc, MathTex(r"\vec{v}^{hxc}"))
        )
        self.wait()
        m_v_ext_main = decimal_matrix(mol1.v_ext.reshape(-1, 1)).move_to([-4, -2, 0])
        m_v_hxc_main = m_v_ext_vec.copy()
        self.play(
            FadeIn(m_v_ext_main),
            Transform(m_v_hxc_main, decimal_matrix(mol1.v_hxc.reshape(-1, 1)).move_to([4, -2, 0]))
        )
        self.wait()
        self.play(
            transform_into(m_v_ext_main, decimal_matrix(np.diag(mol1.v_ext))),
            transform_into(m_v_hxc_main, decimal_matrix(np.diag(mol1.v_hxc))),
            transform_into(text_v_ext, MathTex(r"{V}^{ext}")),
            transform_into(text_v_hxc, MathTex(r"{V}^{Hxc}"))
        )
        self.wait()

        t = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        m_t = decimal_matrix(t).move_to([-3, -1, 0]).scale(0.65)
        text_t = MathTex(r"t").move_to([-3, 1, 0])
        plus1 = MathTex(r"+").move_to([-1, -1, 0])
        plus2 = MathTex(r"+").move_to([3, -1, 0])
        ks_hamiltonian3 = MathTex(r"H ^{KS} = ").move_to([-6, -1, 0])

        self.play(
            FadeIn(ks_hamiltonian3, m_t, plus1, plus2, text_t),
            m_v_hxc_main.animate.move_to([5, -1, 0]).scale(0.65),
            m_v_ext_main.animate.move_to([1, -1, 0]).scale(0.65),
            text_v_ext.animate.move_to([1, 1, 0]),
            text_v_hxc.animate.move_to([5, 1, 0])
        )
        self.wait()

        self.play(
            Transform(text_v_hxc, MathTex(r"{V}^{s}").move_to(text_v_ext)),
            FadeOut(text_v_ext, m_v_ext_main, plus2),
            Transform(m_v_hxc_main, decimal_matrix(np.diag(mol1.v_hxc + mol1.v_ext)).move_to(m_v_ext_main).scale(0.65))
        )
        self.wait()
        self.play(
            Transform(text_v_hxc, MathTex(r"{H}^{KS}").move_to(text_t)),
            FadeOut(text_t, m_t, plus1),
            Transform(m_v_hxc_main, decimal_matrix(mol1.h_ks).move_to(m_t).scale(0.65))
        )
        ks_h_matrix = m_v_hxc_main
        self.wait()
        formula1 = MathTex(r'\hat H^{KS}', r'\ket{\Phi_0}', '=', r'\varepsilon^{KS}_0', r'\ket {\Phi_0}').move_to(
            [3, -1, 0])
        self.play(FadeIn(formula1))
        self.wait(1)
        self.play(FadeOut(ks_hamiltonian3, text_v_hxc, formula1[0]), ks_h_matrix.animate.shift(2 * LEFT))
        question_vector = Matrix(np.array(['?'] * mol1.Ns).reshape(-1, 1)).scale(0.65).next_to(ks_h_matrix, RIGHT, buff=0.5)
        self.play(ReplacementTransform(formula1[1], question_vector, buff=0.5))
        self.play(formula1[2].animate.next_to(question_vector, RIGHT, buff=0.5))
        self.play(formula1[3].animate.next_to(formula1[2], RIGHT, buff=0.5))
        question_vector2 = question_vector.copy().next_to(formula1[3], RIGHT, buff=0.5)
        self.play(ReplacementTransform(formula1[4], question_vector2))
        text1 = Text('Eigenvector problem', font_size=25).move_to([-1, -3, 0])
        temp1 = '<'
        text3 = MathTex(r'\ket{\Phi_' + str(0) + '}' + temp1 + r'N_e/2=' + str(mol1.Ne // 2)).move_to([-2, 1, 0])
        global gamma
        t_gamma = DecimalMatrix(gamma, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
            [4, -2.5, 0])
        text4 = Text('1RDM').move_to([4, -1, 0])
        self.play(FadeIn(text1, text3, t_gamma, text4))
        self.wait(0.5)
        for id1 in range(mol1.Ns):
            if id1 < mol1.Ne // 2:
                temp1 = '<'
            else:
                temp1 = r'\nless'
            self.play(transform_into(question_vector, decimal_matrix(eig_vec[:, id1].reshape(-1, 1))),
                      transform_into(question_vector2, decimal_matrix(eig_vec[:, id1].reshape(-1, 1))),
                      transform_into(formula1[3], Text(f"{eig_val[id1]:5.2f}").scale(0.65)),
                      transform_into(text3, MathTex(r'\ket{\Phi_' + str(id1) + '}' + temp1 + r' N_e/2=' + str(mol1.Ne // 2)))
                      )
            if id1 < mol1.Ne // 2:
                wf_mult = question_vector.copy()
                wf_mult_T = question_vector.copy()
                self.play(wf_mult.animate.move_to([2.5, 0.5, 0]))
                self.play(
                    Transform(wf_mult_T, decimal_matrix(eig_vec[:, id1][np.newaxis]).move_to([5, 0.5, 0]))
                )
                vec_i = eig_vec[:, id1][np.newaxis]
                gamma += vec_i.T @ vec_i
                self.play(transform_into(t_gamma, decimal_matrix(gamma)), FadeOut(wf_mult, wf_mult_T, SHIFT=DOWN))
            self.wait()
        self.wait(3)
        self.play(FadeOut(text1, text3, question_vector, question_vector2, formula1[3], title1, ks_h_matrix))


class test(Scene):
    def construct(self):
        mol1 = MoleculeData(4, 4, param_dict, {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, [[0, 2], [1, 3]])
        self.add(NumberPlane())
        theta = np.pi / 4
        r = 1.5
        points = []
        lines = []
        sites_param = []
        edge_param = []
        for index1, i in enumerate(np.linspace(theta, np.pi * 2 + theta, mol1.Ns, False)):
            points.append(Circle().move_to([r * np.cos(i), r * np.sin(i), 0]).set_color(WHITE).scale(0.3))
            sites_param.append(
                Text(f'U={param_dict[index1]["U"]}, v_ext={param_dict[index1]["v"]}', font_size=23).move_to(
                    [1.5 * r * np.cos(i), 1.5 * r * np.sin(i), 0]))
            if index1 != mol1.Ns - 1:
                edge_param.append(
                    Text(f't={mol1.t_dict[(index1, index1 + 1)]}', font_size=23).move_to(
                        [1.5 * r * np.cos(i + np.pi / mol1.Ns), 1.5 * r * np.sin(i + np.pi / mol1.Ns), 0]))

            if index1 % 2 == 0:
                points[-1].set_fill(WHITE, opacity=1)
            else:
                points[-1].set_fill(BLACK, opacity=1)
        for i in range(len(points) - 1):
            lines.append(Line(points[i].get_center(), points[i + 1].get_center()).set_color(WHITE))
        edge_param.append(
            Text(f't={mol1.t_dict[(len(points) - 1, 0)]}', font_size=23).move_to(
                [1.25 * r * np.cos(theta - np.pi / mol1.Ns), 1.25 * r * np.sin(theta - np.pi / mol1.Ns), 0]))
        lines.append(Line(points[0].get_center(), points[-1].get_center()).set_color(WHITE))
        molecule = Group(*lines, *points)
        text_electrons = Text(f'{mol1.Ne}e', font_size=50)
        molecule = molecule.scale(0.6).move_to([6, 3, 0])
        text_electrons = text_electrons.move_to([6, 3, 0])
        m_v_ext_vec = decimal_matrix([mol1.v_hxc])
        vec1 = MathTex(r"\vec v^{Hxc} = ")
        v_hxc_horizontal = Group(vec1, m_v_ext_vec).arrange(RIGHT, buff=0.5).move_to([0, -1, 0]).move_to(
            [2.9, 3, 0]).scale(0.75)
        y0 = 2
        x0 = 0.5
        separating_line1 = Line([7, y0, 0], [-7, y0, 0])
        separating_line2 = Line([x0, y0, 0], [x0, 4, 0])
        t_h_ks = DecimalMatrix(mol1.h_ks, element_to_mobject_config={"num_decimal_places": 2}).move_to(
            [5, 0.5, 0]).scale(0.65)
        # global gamma
        t_gamma = DecimalMatrix(mol1.y_a, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
            [5, -2.5, 0])
        text_gamma = Tex(r'$\gamma=$').next_to(t_gamma, LEFT)
        text_h_ks = Tex(r'$H^{KS}=$').next_to(t_h_ks, LEFT)
        text1 = Text(' ').move_to([-3, 1.5, 0])
        self.add(separating_line1, separating_line2, v_hxc_horizontal, molecule, text_electrons, t_h_ks,
                 t_gamma,
                 text_h_ks, text_gamma)
        # CASCI
        title1 = Text(r"CASCI", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(FadeIn(title1))
        for impurity_id in range(2):
            self.play(Transform(text1, MarkupText(f'Impurity number:<span fgcolor="{YELLOW}">{impurity_id}</span>',
                                                  font_size=30).move_to([-4.5, 1.5, 0])))
            squared_sites = []
            for i in mol1.equivalent_sites[impurity_id]:
                squared_sites.append(SurroundingRectangle(mobject=molecule[-mol1.Ns + i], color=YELLOW, buff=0.15))
                self.play(Create(squared_sites[-1]))
            gamma2 = mol1.y_a
            if 0 in mol1.equivalent_sites[impurity_id]:
                text2 = Text(f"Impurity is in correct place for Householder transformation.\n"
                             f"We don't need to cahnge indices", font_size=20).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait(1)
            else:
                site_id_old = int(mol1.equivalent_sites[impurity_id][0])
                text2 = Text(f"Impurity is not in correct place for Householder transformation.\n"
                             f"We need to change indices {site_id_old} and 0", font_size=20).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait()
                h_ks2 = mol1.h_ks

                gamma2[:, [0, site_id_old]] = gamma2[:, [site_id_old, 0]]
                h_ks2[:, [0, site_id_old]] = h_ks2[:, [site_id_old, 0]]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={
                    "num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={
                              "num_decimal_places": 2}).scale(0.65)))
                gamma2[[0, site_id_old], :] = gamma2[[site_id_old, 0], :]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={
                    "num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={
                              "num_decimal_places": 2}).scale(0.65)))
            self.play(FadeOut(text2))
            arrow = Arrow([2, -2.5, 0], [0.5, -2.5, 0])
            self.play(Create(arrow))
            mol1.get_access_to_variables_casci(impurity_id)
            t_P = DecimalMatrix(mol1.P, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
                [-1.5, -2.5, 0])
            text_P = Tex('P=').next_to(t_P, LEFT)
            self.play(Create(t_P), FadeIn(text_P))
            arrow2 = Arrow([-1.5, -1, 0], [-1.5, 0, 0])
            eq1 = MathTex(r'\tilde h = P \cdot H^{KS} \cdot P').move_to([-1.5, 0.5, 0])

            self.play(Create(arrow2), FadeIn(eq1))
            arrow3 = Arrow([0.5, 0.5, 0], [1.7, 0.5, 0])

            self.play(Create(arrow3),
                      transform_into(text_h_ks, MathTex(r'\tilde h=')),
                      transform_into(t_h_ks, decimal_matrix(mol1.h_tilde)))
            self.play(FadeOut(arrow, arrow2, t_P, text_P, arrow3, eq1))

            h_tilde_box = SurroundingRectangle(mobject=VGroup(t_h_ks[0][0], t_h_ks[0][mol1.Ns + 1]), color=YELLOW,
                                               buff=0.30)
            self.play(Create(h_tilde_box))
            self.play(FadeOut(h_tilde_box, target_position=decimal_matrix(mol1.h_tilde_dimer).to_edge([1, 0, 0],
                                                                                                      buff=0.30)),
                      Transform(t_h_ks, decimal_matrix(mol1.h_tilde_dimer).to_edge([1, 0, 0], buff=0.15)))
            self.play(Transform(text_h_ks, MathTex(r'\tilde h^{dimer}=').next_to(t_h_ks, LEFT)))

            eq1 = MathTex(r'\hat H^{(' + str(impurity_id) + r')} =\sum_\sigma \sum_{pq}^1 \tilde h_{pq} '
                          r'\hat d^{\dagger}_{p\sigma}d^{ }_{q\sigma} + \sum_\sigma \mu^{imp}\hat d^{\dagger}_{p\sigma}'
                          r'd^{ }_{q\sigma} + U \hat n_{0\uparrow}\hat n_{0\downarrow}').\
                move_to([-1.5, 0.5, 0]).scale(0.6)
            self.play(Create(eq1))

            grid, grid_labels = generate_empty_graph(mol1.dimer_opt_list)
            grid.move_to([-3, -2, 0])
            dots = VGroup(*[Dot(point=grid.c2p(coord[0], coord[1], 0), radius=0.05) for coord in mol1.dimer_opt_list])
            min_mu_imp = np.min(mol1.dimer_opt_list[:, 0])
            max_mu_imp = np.max(mol1.dimer_opt_list[:, 0])
            desired_density = mol1.n_ks[0]
            line_desired_density = Line(grid.c2p(min_mu_imp, desired_density, 0),
                                        grid.c2p(max_mu_imp, desired_density, 0))

            self.play(FadeIn(grid))
            self.play(ReplacementTransform(t_gamma[0][0].copy(), line_desired_density))
            self.play(ShowIncreasingSubsets(dots, run_time=3.0))

            lines_to_grid = grid.get_lines_to_point(grid.c2p(mol1.dimer_opt_list[-1, 0], mol1.dimer_opt_list[-1, 1]))[1]
            self.play(Create(lines_to_grid))
            text_mu_imp1 = DecimalNumber(mol1.mu_imp).next_to(grid.c2p(mol1.dimer_opt_list[-1, 0], 0), DOWN, buff=0.5)
            text_mu_imp2 = MathTex(r'=\mu^{imp}\longrightarrow v^{Hxc}').next_to(text_mu_imp1, LEFT, 0.15)
            self.play(FadeIn(text_mu_imp1, text_mu_imp2))
            self.wait(1)
            # self.play(ReplacementTransform(text_mu_imp, ))
            self.play(FadeOut(text_mu_imp2), ReplacementTransform(text_mu_imp1, m_v_ext_vec[0][impurity_id]))
            self.play(*[m_v_ext_vec[0][id].animate.set_value(mol1.mu_imp) for id in mol1.equivalent_sites[impurity_id]])
            self.play(FadeOut(t_gamma, dots, grid, eq1, lines_to_grid, line_desired_density))
            self.play(FadeOut(*squared_sites, text_h_ks),
                      Transform(t_h_ks, decimal_matrix(mol1.h_ks).move_to([5, 0.5, 0])))
            text_h_ks = MathTex(r'H^{KS}=').next_to(t_h_ks, LEFT)
            self.play(FadeIn(text_h_ks))


def generate_empty_graph(input_array):
    delta_x = np.max(input_array[:, 0]) - np.min(input_array[:, 0])
    delta_y = np.max(input_array[:, 0]) - np.min(input_array[:, 0])
    x_range = [np.min(input_array[:, 0]) - 0.2 * delta_x, np.max(input_array[:, 0]) + 0.2 * delta_x, 0.2]
    y_range = [np.min(input_array[:, 1]) - 0.2 * delta_y, np.max(input_array[:, 1]) + 0.2 * delta_y, 0.2]
    grid = Axes(

        x_range=x_range,
        # step size determines num_decimal_places.
        y_range=y_range,
        x_length=3,
        y_length=3,
        x_axis_config={
            "numbers_to_include": np.arange(*x_range),
            "font_size": 20,
        },
        y_axis_config={
            "numbers_to_include": np.arange(*y_range),
            "font_size": 20,
        },
        tips=False,
    )

    # Labels for the x-axis and y-axis.
    y_label = grid.get_y_axis_label("y", edge=LEFT, direction=LEFT, buff=0.4)
    x_label = grid.get_x_axis_label("x", edge=DOWN, direction=DOWN, buff=0.4)
    grid_labels = VGroup(x_label, y_label)

    # dots = VGroup(*[Dot(point=grid.c2p(coord[0], coord[1], 0)) for coord in points])
    # self.add(dots)
    return grid, grid_labels

class test2(Scene):
    def construct(self):
        m_v_ext_vec = decimal_matrix([np.array([0,0,0,0])])
        indices = [0, 1]
        self.add(m_v_ext_vec)
        self.play(*[m_v_ext_vec[0][id1].animate.set_value(3) for id1 in indices])


if __name__ == "__main__":
    from subprocess import call
    call(["python", "-m", "manim", "-p", "-ql", "generate_animation.py", "test"])
