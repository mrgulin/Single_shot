import Quant_NBody
import LPFET.lpfet as lpfet
from manim import *

dimer_opt_list = []


def cost_function_CASCI(mu_imp, embedded_mol, h_tilde_dimer, u_0_dimer, desired_density):
    global dimer_opt_list
    mu_imp = mu_imp[0]
    mu_imp_array = np.array([[mu_imp, 0], [0, 0]])
    embedded_mol.build_hamiltonian_fermi_hubbard(h_tilde_dimer - mu_imp_array, u_0_dimer)
    embedded_mol.diagonalize_hamiltonian()
    density_dimer = embedded_mol.calculate_1rdm(index=0)
    dimer_opt_list.append([mu_imp, density_dimer[0, 0], desired_density])
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
        self.h_tilde_dimer_current = np.array([])
        self.dimer_opt_list = np.array([])
        self.mu_imp = -10

    def get_access_to_variables_casci(self, site_id):
        global dimer_opt_list

        # Householder transforms impurity on index 0 so we have to make sure that impurity is on index 0:
        y_a_correct_imp = lpfet.change_indices(self.y_a, site_id)
        t_correct_imp = lpfet.change_indices(self.t, site_id)
        v_s_correct_imp = lpfet.change_indices(self.v_s, site_id)

        print(f'input side_id: {site_id}', v_s_correct_imp, self.v_s)

        self.P, self.v = Quant_NBody.householder_transformation(y_a_correct_imp)
        self.h_tilde = self.P @ (t_correct_imp + np.diag(v_s_correct_imp)) @ self.P

        h_tilde_dimer = self.h_tilde[:2, :2]
        u_0_dimer = np.zeros((2, 2, 2, 2), dtype=np.float64)
        u_0_dimer[0, 0, 0, 0] += self.u[site_id]
        # h_tilde_dimer[0,0] += self.v_ext[site_id]
        mu_imp = self.v_hxc[[site_id]]  # Double parenthesis so I keep array, in future this will be list of
        # indices for block householder

        self.h_tilde_dimer_current = h_tilde_dimer
        opt_v_imp_obj = lpfet.sc_opt.minimize(cost_function_CASCI, mu_imp,
                                              args=(self.embedded_mol, h_tilde_dimer, u_0_dimer, self.n_ks[site_id]),
                                              method='BFGS', options={'eps': 1e-5})
        # This minimize cost function (difference between KS occupations and CASCI occupations squared)
        error = opt_v_imp_obj['fun']
        mu_imp = opt_v_imp_obj['x'][0]

        self.dimer_opt_list = np.array(dimer_opt_list)
        dimer_opt_list = []
        self.mu_imp = mu_imp


v_ext_amplitude = 1
param_dict = dict()
for i in range(4):
    param_dict[i] = {'v': v_ext_amplitude * (-1) ** i, 'U': 3}


def transform_into(old_object, new_object):
    return Transform(old_object, new_object.move_to(old_object.get_center()))



def generate_empty_graph(input_array):
    # delta_x = np.max(input_array[:, 0]) - np.min(input_array[:, 0])
    # delta_y = np.max(input_array[:, 0]) - np.min(input_array[:, 0])
    # x_range = [np.min(input_array[:, 0]) - 0.2 * delta_x, np.max(input_array[:, 0]) + 0.2 * delta_x, 0.2]
    # y_range = [np.min(input_array[:, 1]) - 0.2 * delta_y, np.max(input_array[:, 1]) + 0.2 * delta_y, 0.2]
    x_range = [np.min(input_array[:, 0]), np.max(input_array[:, 0]),
               10 ** (np.round(np.log10(
                   abs(round(np.min(input_array[:, 0]), 3) - round(np.max(input_array[:, 0]), 3)) / 3 * 20))) / 20]
    y_range = [np.min(input_array[:, 1]), np.max(input_array[:, 1]),
               10 ** (np.round(np.log10(
                   abs(round(np.min(input_array[:, 1]), 3) - round(np.max(input_array[:, 1]), 3)) / 3 * 20))) / 20]
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
    grid_labels = grid.get_axis_labels(x_label="u imp", y_label="n")

    return grid, grid_labels

class GenerateRing(Scene):
    def __init__(
            self,
            renderer=None,
            camera_class=Camera,
            always_update_mobjects=False,
            random_seed=None,
            skip_animations=False,
    ):
        super().__init__(renderer, camera_class, always_update_mobjects, random_seed, skip_animations)

        mol1 = MoleculeData(4, 4, param_dict, {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}, [[0, 2], [1, 3]])
        mol1.density_progress.append(mol1.n_ks.copy())
        mol1.casci(0)
        mol1.v_hxc_progress.append(mol1.v_hxc.copy())
        mol1.calculate_ks()
        self.old_density = mol1.n_ks
        self.mol1 = mol1
        self.m_v_hxc_bar = None
        self.molecule_scheme = None


    def build_molecule_animation(self):
        mol1 = self.mol1
        theta = np.pi / 4
        r = 1.5
        points = []
        lines = []
        sites_param = []
        edge_param = []
        for index1, i in enumerate(np.linspace(theta, np.pi * 2 + theta, mol1.Ns, False)):
            points.append(Circle().move_to([r * np.cos(i), r * np.sin(i), 0]).set_color(WHITE).scale(0.3))
            sites_param.append(
                MathTex(f'U={param_dict[index1]["U"]}, v_{{ext}}={param_dict[index1]["v"]:+.1f}', font_size=23).move_to(
                    [1.5 * r * np.cos(i), 1.5 * r * np.sin(i), 0]))
            if index1 != mol1.Ns - 1:
                edge_param.append(
                    MathTex(f't={mol1.t_dict[(index1, index1 + 1)]}', font_size=23).move_to(
                        [1.25 * r * np.cos(i + np.pi / mol1.Ns), 1.25 * r * np.sin(i + np.pi / mol1.Ns), 0]))

            if index1 % 2 == 0:
                points[-1].set_fill(WHITE, opacity=1)
            else:
                points[-1].set_fill(BLACK, opacity=1)
        for i in range(len(points) - 1):
            lines.append(Line(points[i].get_center(), points[i + 1].get_center()).set_color(WHITE))
        edge_param.append(
            MathTex(f't={mol1.t_dict[(len(points) - 1, 0)]}', font_size=23).move_to(
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

        v_hxc_horizontal = [mol1.v_hxc]
        m_v_hxc_vec = decimal_matrix(v_hxc_horizontal)

        title = Tex(r"Starting Hxc approximation").move_to([0, 2, 0])
        text_explanation_hxc = Text(
            "Usually we start with approximation that Hxc potential is equal\n"
            "to zero on all sites. In this presentation I will show second \n"
            "iteration where Hxc potentials are already different from zero.", font_size=20).move_to([0, 0.25, 0])
        vec1 = MathTex(r"\vec v^{Hxc} = ")
        v_hxc_horizontal = Group(vec1, m_v_hxc_vec).arrange(RIGHT, buff=0.2, aligned_edge=DOWN).move_to([0, -2, 0])
        self.play(FadeIn(title, v_hxc_horizontal, text_explanation_hxc))
        self.wait(4)
        self.play(FadeOut(title, title1, text_explanation_hxc), v_hxc_horizontal.animate.move_to([2.7, 3, 0]).scale(0.95))
        y0 = 2
        x0 = 0
        separating_line1 = Line([7, y0, 0], [-7, y0, 0])
        separating_line2 = Line([x0, y0, 0], [x0, 4, 0])
        self.play(Create(separating_line1))
        self.play(Create(separating_line2))
        self.wait()
        self.m_v_hxc_bar = v_hxc_horizontal
        self.molecule_scheme = molecule


    def build_molecule_static(self):
        self.add(NumberPlane())
        mol1 = self.mol1
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
        m_v_hxc_vec = decimal_matrix([mol1.v_hxc])
        vec1 = MathTex(r"\vec v^{Hxc} = ")
        v_hxc_horizontal = Group(vec1, m_v_hxc_vec).arrange(RIGHT, buff=0.5).move_to([0, -1, 0]).move_to(
            [2.7, 3, 0])
        y0 = 2
        x0 = 0.5
        separating_line1 = Line([7, y0, 0], [-7, y0, 0])
        separating_line2 = Line([x0, y0, 0], [x0, 4, 0])
        self.m_v_hxc_bar = v_hxc_horizontal
        self.molecule_scheme = molecule
        self.add(separating_line1, separating_line2, v_hxc_horizontal, molecule, text_electrons)

    def animate_ks(self):
        mol1 = self.mol1

        title1 = Text(r"Kohn-Sham system", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(Write(title1))

        title = Tex(r"KS Hamiltonian").move_to([0, 0.5, 0])
        ks_hamiltonian = MathTex(r"\hat H ^{KS} = \hat T + ", r"\hat V^{s}", " ", r" ").move_to([0, -1.5, 0])
        self.play(
            Write(title),
            FadeIn(ks_hamiltonian)
        )
        self.wait()
        ks_hamiltonian2 = MathTex(r"\hat H ^{KS} = \hat T + ", r"\hat V^{ext}", "+", r"\hat V^{Hxc}").move_to(
            [0, -1.5, 0])
        self.play(
            TransformMatchingTex(ks_hamiltonian, ks_hamiltonian2)
        )
        self.wait()
        ks_hamiltonian = ks_hamiltonian2
        self.play(
            FadeOut(title, ks_hamiltonian[0], ks_hamiltonian[2]),
            ks_hamiltonian[1].animate.move_to([-4, 0, 0]),
            ks_hamiltonian[3].animate.move_to([4, 0, 0])
        )
        self.wait()

        # Matrix calculation of KS Hamiltonian
        text_v_ext = ks_hamiltonian[1]
        text_v_hxc = ks_hamiltonian[3]

        self.play(
            transform_into(text_v_ext, MathTex(r"\vec{v}^{ext}")),
            transform_into(text_v_hxc, MathTex(r"\vec{v}^{hxc}"))
        )
        m_v_ext_main = decimal_matrix(mol1.v_ext.reshape(-1, 1)).move_to([-4, -2, 0])
        m_v_hxc_main = self.m_v_hxc_bar[1].copy()
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

        t = mol1.t
        ks_hamiltonian3 = MathTex(r"H ^{KS} = ").move_to([-6.1, -1, 0])
        m_t = decimal_matrix(t).move_to([-3.1, -1, 0])
        text_t = MathTex(r"t").move_to([-3.1, 1, 0])
        plus1 = MathTex(r"+").move_to([-1, -1, 0]).scale(0.8)
        plus2 = MathTex(r"+").move_to([3, -1, 0]).scale(0.8)


        self.play(
            FadeIn(ks_hamiltonian3, m_t, plus1, plus2, text_t),
            m_v_hxc_main.animate.move_to([5, -1, 0]),
            m_v_ext_main.animate.move_to([1, -1, 0]),
            text_v_ext.animate.move_to([1, 1, 0]),
            text_v_hxc.animate.move_to([5, 1, 0])
        )
        self.wait()

        self.play(
            Transform(text_v_hxc, MathTex(r"{V}^{s}").move_to(text_v_ext)),
            FadeOut(text_v_ext, m_v_ext_main, plus2),
            Transform(m_v_hxc_main, decimal_matrix(np.diag(mol1.v_hxc + mol1.v_ext)).move_to(m_v_ext_main))
        )
        self.wait()
        self.play(
            Transform(text_v_hxc, MathTex(r"{H}^{KS}").move_to(text_t)),
            FadeOut(text_t, m_t, plus1),
            Transform(m_v_hxc_main, decimal_matrix(mol1.h_ks).move_to(m_t))
        )
        ks_h_matrix = m_v_hxc_main
        self.wait()
        formula1 = MathTex(r'\hat H^{KS}', r'\ket{\Phi_0}', '=', r'\varepsilon^{KS}_0', r'\ket {\Phi_0}').move_to(
            [3, -1, 0])
        self.play(FadeIn(formula1))
        self.wait(1)
        self.play(FadeOut(ks_hamiltonian3, text_v_hxc, formula1[0]), ks_h_matrix.animate.shift(2 * LEFT))
        question_vector = Matrix(np.array(['?'] * mol1.Ns).reshape(-1, 1)).scale(0.65).next_to(ks_h_matrix, RIGHT,
                                                                                               buff=0.5)
        self.play(ReplacementTransform(formula1[1], question_vector, buff=0.5))
        self.play(formula1[2].animate.next_to(question_vector, RIGHT, buff=0.5))
        self.play(formula1[3].animate.next_to(formula1[2], RIGHT, buff=0.5))
        question_vector2 = question_vector.copy().next_to(formula1[3], RIGHT, buff=0.5)
        self.play(ReplacementTransform(formula1[4], question_vector2))
        text1 = Text('Eigenvector problem', font_size=25).move_to([-1, -3, 0])
        temp1 = '<'
        text3 = MathTex(r'\ket{\Phi_' + str(0) + '}:' + str(0) + temp1 + r'N_e/2=' + str(mol1.Ne // 2)).move_to([-2, 1, 0])
        gamma = np.zeros((mol1.Ns, mol1.Ns))
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
            self.play(transform_into(question_vector, decimal_matrix(mol1.wf_ks[:, id1].reshape(-1, 1))),
                      transform_into(question_vector2, decimal_matrix(mol1.wf_ks[:, id1].reshape(-1, 1))),
                      transform_into(formula1[3], Text(f"{mol1.epsilon_s[id1]:5.2f}").scale(0.65)),
                      transform_into(text3,
                                     MathTex(r'\ket{\Phi_' + str(id1) + '}:' + str(id1) + temp1 + r' N_e/2=' + str(mol1.Ne // 2)))
                      )
            if id1 < mol1.Ne // 2:
                wf_mult = question_vector.copy()
                wf_mult_T = question_vector.copy()
                self.play(wf_mult.animate.move_to([2.5, 0.5, 0]))
                self.play(
                    Transform(wf_mult_T, decimal_matrix(mol1.wf_ks[:, id1][np.newaxis]).move_to([5, 0.5, 0]))
                )
                vec_i = mol1.wf_ks[:, id1][np.newaxis]
                gamma += vec_i.T @ vec_i
                self.play(transform_into(t_gamma, decimal_matrix(gamma)),
                          FadeOut(wf_mult, target_position=t_gamma),
                          FadeOut(wf_mult_T, target_position=t_gamma))
            self.wait()
        self.wait(3)
        self.play(FadeOut(text1, text3, question_vector, question_vector2, formula1[3], formula1[2],
                          title1, ks_h_matrix, t_gamma, text4))


    def casci(self):
        mol1 = self.mol1
        t_gamma = DecimalMatrix(mol1.y_a, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
            [5, -2.5, 0])
        t_h_ks = DecimalMatrix(mol1.h_ks, element_to_mobject_config={"num_decimal_places": 2}).move_to(
            [5, 0.5, 0]).scale(0.65)
        text_gamma = Tex(r'$\gamma=$').next_to(t_gamma, LEFT)
        text_h_ks = Tex(r'$H^{KS}=$').next_to(t_h_ks, LEFT)
        self.play(FadeIn(t_h_ks, t_gamma, text_h_ks, text_gamma))

        title1 = Text(r"CASCI", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(FadeIn(title1))
        first = True
        text1 = None
        for impurity_id in range(2):
            if not first:
                t_h_ks = DecimalMatrix(mol1.h_ks, element_to_mobject_config={"num_decimal_places": 2}).move_to(
                    [5, 0.5, 0]).scale(0.65)
                # global gamma
                t_gamma = DecimalMatrix(mol1.y_a, element_to_mobject_config={"num_decimal_places": 2}).scale(
                    0.65).move_to(
                    [5, -2.5, 0])
                text_gamma = Tex(r'$\gamma=$').next_to(t_gamma, LEFT)
                text_h_ks = Tex(r'$H^{KS}=$').next_to(t_h_ks, LEFT)
                self.play(Transform(text1, MarkupText(f'Impurity number:<span fgcolor="{YELLOW}">{impurity_id}</span>',
                                                      font_size=30).move_to([-4.5, 1.5, 0])),
                          FadeIn(t_h_ks, t_gamma, text_gamma, text_h_ks))
            else:
                text1 = MarkupText(f'Impurity number:<span fgcolor="{YELLOW}">{impurity_id}</span>',
                                   font_size=30).move_to([-4.5, 1.5, 0])
                self.play(FadeIn(text1))
            squared_sites = []
            for i in mol1.equivalent_sites[impurity_id]:
                squared_sites.append(SurroundingRectangle(mobject=self.molecule_scheme[-mol1.Ns + i], color=YELLOW, buff=0.15))
                self.play(Create(squared_sites[-1]))
            gamma2 = mol1.y_a.copy()
            if 0 in mol1.equivalent_sites[impurity_id]:
                text2 = Text(f"Impurity  is  in  correct  place  for  Householder  transformation.\n"
                             f"We  don't  need  to  change  indices", font_size=20,
                             t2c={" is ": YELLOW}).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait(1)
            else:
                site_id_old = int(mol1.equivalent_sites[impurity_id][0])
                text2 = Text(f"Impurity  is  not  in  correct  place  for  Householder  transformation.\n"
                             f"We  need  to  change  indices  {site_id_old}  and  0", font_size=20,
                             t2c={" is  not ": YELLOW, f" {site_id_old} ": YELLOW}).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait()
                h_ks2 = mol1.h_ks.copy()

                gamma2[:, [0, site_id_old]] = gamma2[:, [site_id_old, 0]]
                h_ks2[:, [0, site_id_old]] = h_ks2[:, [site_id_old, 0]]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={
                    "num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={
                              "num_decimal_places": 2}).scale(0.65)))
                gamma2[[0, site_id_old], :] = gamma2[[site_id_old, 0], :]
                h_ks2[[0, site_id_old], :] = h_ks2[[site_id_old, 0], :]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={
                    "num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={
                              "num_decimal_places": 2}).scale(0.65)))
            self.play(FadeOut(text2))
            arrow = Arrow([2, -2.5, 0], [0.5, -2.5, 0])
            self.play(Create(arrow))
            mol1.get_access_to_variables_casci(impurity_id)
            print(f'impurity id: {impurity_id}, ')
            t_P = DecimalMatrix(mol1.P, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
                [-1.5, -2.5, 0])
            text_P = Tex('P=').next_to(t_P, LEFT)
            self.play(Create(t_P), FadeIn(text_P))
            eq1 = MathTex(r"\tilde h = P \cdot h' \cdot P").move_to([-1.5, 0.5, 0])
            eq2 = MathTex(r"\tilde \gamma = P \cdot \gamma' \cdot P").next_to(eq1, DOWN, buff=0.1)
            arrow2 = Arrow([-1.5, -1.7, 0], [-1.5, -0.2, 0])
            self.play(Create(arrow2), Write(eq1), Write(eq2))
            arrow3 = Arrow([0.5, 0.5, 0], [1.7, 0.5, 0])
            arrow4 = Arrow(eq2, text_gamma)
            gamma2 = mol1.P @ gamma2 @ mol1.P
            self.play(Create(arrow3), Create(arrow4),
                      transform_into(text_h_ks, MathTex(r'\tilde h=')),
                      transform_into(t_h_ks, decimal_matrix(mol1.h_tilde)),
                      transform_into(text_gamma, MathTex(r'\tilde \gamma=')),
                      transform_into(t_gamma, decimal_matrix(gamma2)))
            self.wait(3)
            self.play(FadeOut(arrow, arrow2, arrow3, arrow4, eq1, eq2, t_P, text_P))

            h_tilde_box = SurroundingRectangle(mobject=VGroup(t_h_ks[0][0], t_h_ks[0][mol1.Ns + 1]), color=YELLOW,
                                               buff=0.30)
            self.play(Create(h_tilde_box))
            self.play(FadeOut(h_tilde_box, target_position=decimal_matrix(mol1.h_tilde_dimer_current).to_edge([1, 0, 0],
                                                                                                      buff=0.30)),
                      Transform(t_h_ks, decimal_matrix(mol1.h_tilde_dimer_current).to_edge([1, 0, 0], buff=0.15)),
                      Transform(text_h_ks, MathTex(r'\tilde h^{dimer}=').move_to([3.8, 0, 0])))

            eq1 = MathTex(r'\hat H^{(' + str(impurity_id) + r')} =\sum_\sigma \sum_{pq}^1 \tilde h_{pq} '
                                                            r'\hat d^{\dagger}_{p\sigma}d^{ }_{q\sigma} - \sum_\sigma \mu^{imp}\hat d^{\dagger}_{0\sigma}'
                                                            r'd^{ }_{0\sigma} + U \hat n_{0\uparrow}\hat n_{0\downarrow}'). \
                move_to([-1.5, 0.5, 0]).scale(0.6)
            self.play(Create(eq1))
            grid, grid_labels = generate_empty_graph(mol1.dimer_opt_list)
            grid_x_label = MathTex('\mu^{imp}').scale(0.8).move_to([-2.1, -3.05, 0])
            grid_y_label = MathTex('n').scale(0.8).move_to([-6, 0, 0])
            grid.move_to([-4.5, -2, 0])
            dots = VGroup(*[Dot(point=grid.c2p(coord[0], coord[1], 0), radius=0.05,
                                color=YELLOW) for coord in mol1.dimer_opt_list])
            min_mu_imp = np.min(mol1.dimer_opt_list[:, 0])
            max_mu_imp = np.max(mol1.dimer_opt_list[:, 0])
            desired_density = gamma2[0, 0]
            print(min_mu_imp, max_mu_imp, desired_density, grid.c2p(min_mu_imp, desired_density, 0),  grid.c2p(max_mu_imp, desired_density, 0))
            line_desired_density = Line(grid.c2p(min_mu_imp, desired_density, 0),
                                        grid.c2p(max_mu_imp, desired_density, 0))

            self.play(FadeIn(grid, grid_x_label, grid_y_label))
            self.play(ReplacementTransform(t_gamma[0][0].copy(), line_desired_density))
            self.play(ShowIncreasingSubsets(dots, run_time=4.0))

            lines_to_grid = grid.get_lines_to_point(grid.c2p(mol1.dimer_opt_list[-1, 0], mol1.dimer_opt_list[-1, 1]))[1]
            self.play(Create(lines_to_grid))
            text_mu_imp1 = DecimalNumber(mol1.mu_imp).move_to([-2, -2, 0]).scale(0.8)
            text_mu_imp2 = MathTex(r'=\mu^{imp}\longrightarrow v^{Hxc}').move_to([0, -2, 0]).scale(0.75)
            self.play(FadeIn(text_mu_imp1, text_mu_imp2))
            self.wait(1)
            self.play(FadeOut(text_mu_imp2), ReplacementTransform(text_mu_imp1, self.m_v_hxc_bar[1][0][impurity_id]))
            self.play(*[self.m_v_hxc_bar[1][0][id].animate.set_value(mol1.mu_imp) for id in mol1.equivalent_sites[impurity_id]])
            self.play(FadeOut(t_gamma, dots, grid, grid_x_label, grid_y_label, eq1, lines_to_grid, line_desired_density,
                              *squared_sites, text_h_ks, t_h_ks, text_gamma))
            first = False
        self.play(FadeOut(title1, text1))
        # endregion

    def construct(self):
        self.build_molecule_animation()
        self.animate_ks()
        self.casci()
        self.continuation()

    def construct2(self):
        self.build_molecule_static()
        self.continuation()

    def continuation(self):
        mol1 = self.mol1

        # region First screen
        title1 = Text(r"Continuation", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        self.play(FadeIn(title1))

        text1 = Text("Now we have generated new approximations for Hxc\n"
                     "potential and we are able to calculate KS  orbitals.\n "
                     "This means that we loop between KS system and CASCI\n "
                     "and after each iteration we generate new \n"
                     "approximation for the Hxc potentials and we do this\n"
                     "until occupation numbers converge.",
                     font_size=20).move_to([-3, 0.5, 0])
        self.play(FadeIn(text1))
        self.wait(4)
        ax = Axes(x_range=[0, 16.5], y_range=[0, 3.2], x_length=7, y_length=4,
                  x_axis_config={"numbers_to_include": np.arange(0, 16.1, 2), "font_size": 20},
            y_axis_config={"numbers_to_include": np.arange(0, 3.1, 1), "font_size": 20},tips=False)
        ax.move_to([-3, -1.5, 0])
        text_x_label = Text('Iteration').scale(0.5).move_to([1, -3.7, 0])
        text_y_label = MathTex(r'v^{Hxc}').scale(0.7).move_to([-6.5, 1, 0])
        grid_labels = VGroup(text_x_label, text_y_label)

        iter_start = 2
        text_density_sq_text = MathTex(
            r'\sum_i \left( n_i^{(' + str(iter_start) + r')}  -  n_i^{(' + str(iter_start - 1) + r')} \right)/N_s=').scale(
            0.8)
        text_density_sq_num = DecimalNumber(-1).move_to([2.7, 0.1, 0])

        text_density_sq = VGroup(text_density_sq_text, text_density_sq_num).move_to([5, 1, 0]).scale(0.8)
        text_epsilon = MathTex(r'\varepsilon=1 \cdot 10^{-5}').next_to(text_density_sq, DOWN, 0.1).scale(0.8)

        t_gamma = DecimalMatrix(mol1.y_a, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to(
            [5, -2.5, 0])
        text_gamma = Tex(r'$\gamma=$').next_to(t_gamma, LEFT)
        self.play(FadeOut(text1), FadeIn(t_gamma, text_gamma, text_density_sq, text_epsilon, ax, grid_labels))

        # endregion
        # region Run proper CASCI function so I can plot points
        mol1.density_progress.append(mol1.n_ks.copy())
        mol1.casci(0)
        mol1.v_hxc_progress.append(mol1.v_hxc.copy())

        # endregion
        # region Make graph of first points
        dot_list = [VGroup(), VGroup()]
        line_list = [VGroup(), VGroup()]
        colors = [RED, YELLOW]
        for point_site in range(2):
            dot_list[point_site] += Dot(point=ax.c2p(0, 0, 0), radius=0.05, color=YELLOW)
            dot_list[point_site] += Dot(point=ax.c2p(1, mol1.v_hxc_progress[point_site][0], 0), radius=0.05, color=colors[point_site])
            line_list[point_site] += Line(dot_list[point_site][-2], dot_list[point_site][-1], color=colors[point_site])
            dot_list[point_site] += Dot(point=ax.c2p(2, mol1.v_hxc_progress[point_site][1], 0), radius=0.05,
                                        color=colors[point_site])
            line_list[point_site] += Line(dot_list[point_site][-2], dot_list[point_site][-1], color=colors[point_site])
        self.play(Create(dot_list[0][0]), Create(dot_list[1][0]))
        self.play(Create(line_list[0][0]), Create(line_list[1][0]), Create(dot_list[0][1]), Create(dot_list[1][1]))
        self.wait(2)
        text_iteration = Text(f'Iteration {2}', font_size=25).move_to([-5, 1.5, 0])
        sq_diff_val = np.average(np.square(mol1.n_ks - self.old_density))
        self.play(Transform(self.m_v_hxc_bar[1][0].copy(), dot_list[0][2]),
                  Transform(self.m_v_hxc_bar[1][1].copy(), dot_list[1][2]),
                  Create(line_list[0][1]), Create(line_list[1][1]),
                  text_density_sq_num.animate.set_value(sq_diff_val),
                  FadeIn(text_iteration)
                  )
        self.old_density = mol1.n_ks
        # endregion
        # region Loop
        i = 3
        for i in range(3, 8):
            mol1.calculate_ks()
            mol1.density_progress.append(mol1.n_ks.copy())
            mol1.casci(0)
            mol1.v_hxc_progress.append(mol1.v_hxc.copy())
            for point_site in range(2):
                dot_list[point_site] += Dot(point=ax.c2p(i, mol1.v_hxc_progress[-1][point_site], 0), radius=0.05,
                                            color=colors[point_site])
                line_list[point_site] += Line(dot_list[point_site][-2], dot_list[point_site][-1],
                                              color=colors[point_site])


            sq_diff_val = np.average(np.square(mol1.n_ks - self.old_density))
            self.play(transform_into(text_iteration, Text(f'Iteration {i}', font_size=25)),
                      transform_into(t_gamma, decimal_matrix(mol1.y_a)))
            self.play(transform_into(self.m_v_hxc_bar[1], decimal_matrix([mol1.v_hxc])))
            self.play(Create(line_list[0][i - 1]), Create(line_list[1][i - 1]),
                      Transform(self.m_v_hxc_bar[1][0][0].copy(), dot_list[0][i]),
                      Transform(self.m_v_hxc_bar[1][0][1].copy(), dot_list[1][i]),
                      text_density_sq_num.animate.set_value(sq_diff_val)
                      )
            self.old_density = mol1.n_ks
            self.wait(1)

        text_convergence = "We can see that Hxc potentials alterante\n " \
                           "between two values and they are not going\n " \
                           "to converge by themselves."
        text_convergence = Text(text_convergence, font_size=20).move_to([-0.2, 1, 0])
        #                                             0         1         2                 3               4                                     5
        text_convergence_second = MathTex(r'v_j^{Hxc (i+1)}', '=', r'\mu_j^{imp, (i-1)}', r'+ arctan(', r'\mu_j^{imp, (i)} - \mu_j^{imp, (i-1)}',')').next_to(text_convergence, DOWN, buff=0.2).scale(0.45)
        text_convergence_g = VGroup(text_convergence, text_convergence_second)

        self.play(FadeIn(text_convergence_g))
        # region resize point
        mu_point = dot_list[0][i]
        mu_minus_1_point = dot_list[0][i-1]
        position_mu_minus_1 = mu_minus_1_point.get_center()
        position_mu_minus_1[0] = mu_point.get_center()[0]
        position_mu = mu_point.get_center()
        print(mol1.v_hxc_progress)
        double_arrow = DoubleArrow(position_mu, position_mu_minus_1, buff=0, max_tip_length_to_length_ratio=0.05)
        text_delta = MathTex('\Delta v=').next_to(double_arrow, RIGHT).scale(0.6)
        delta_old_number = mol1.v_hxc_progress[-1][0]-mol1.v_hxc_progress[-2][0]
        delta_old = DecimalNumber(delta_old_number).next_to(text_delta, RIGHT).scale(0.6)
        mu_minus_1_number = mol1.v_hxc_progress[-2][0]
        text_mu_minus_1 = DecimalNumber(mu_minus_1_number).scale(0.6)
        self.play(Create(double_arrow))
        self.play(FadeIn(delta_old, text_delta))
        self.play(Transform(text_convergence_second[2], text_mu_minus_1.move_to(text_convergence_second[2])),
                  Transform(text_convergence_second[4], delta_old.copy().move_to(text_convergence_second[4])))
        result = mu_minus_1_number + np.arctan(delta_old_number)
        delta_new = DecimalNumber(result).scale(0.6).move_to(text_convergence_second[0])
        self.play(Transform(text_convergence_second[0], delta_new))
        def update_arrow(arrow123):
            updated_arrow = DoubleArrow(mu_point, arrow123.end, buff=0, max_tip_length_to_length_ratio=0.05)
            arrow123.become(updated_arrow)
        self.play(FadeOut(text_convergence_second[0].copy(), target_position=mu_point), delta_old.animate.set_value(result),
                  mu_point.animate.move_to(ax.c2p(i, result, 0)),
                  UpdateFromFunc(double_arrow, update_arrow),
                  delta_old.animate.set_value(np.arctan(delta_old_number)))
        v_hxc2 = mol1.v_hxc_progress[-2][1] + np.arctan(mol1.v_hxc_progress[-1][1]-mol1.v_hxc_progress[-2][1])
        self.play(FadeOut(text_convergence_g, double_arrow, text_delta, delta_old))
        self.play(dot_list[1][i].animate.move_to(ax.c2p(i, v_hxc2, 0)))


        mol1.v_hxc = np.array([result, v_hxc2, result, v_hxc2])

        text_with_osc_compensation = Text("With oscillation compensaition", font_size=20).move_to([-1, 1.5, 0])
        self.play(FadeIn(text_with_osc_compensation))
        # endregion

        for i in range(8, 15):
            mol1.calculate_ks()
            mol1.density_progress.append(mol1.n_ks.copy())
            mol1.casci(5)
            mol1.v_hxc_progress.append(mol1.v_hxc.copy())
            for point_site in range(2):
                dot_list[point_site] += Dot(point=ax.c2p(i, mol1.v_hxc_progress[-1][point_site], 0), radius=0.05,
                                            color=colors[point_site])
                line_list[point_site] += Line(dot_list[point_site][-2], dot_list[point_site][-1],
                                              color=colors[point_site])


            sq_diff_val = np.average(np.square(mol1.n_ks - self.old_density))
            self.play(transform_into(text_iteration, Text(f'Iteration {i}', font_size=25)),
                      transform_into(t_gamma, decimal_matrix(mol1.y_a)))
            self.play(transform_into(self.m_v_hxc_bar[1], decimal_matrix([mol1.v_hxc])))
            self.play(Create(line_list[0][i - 1]), Create(line_list[1][i - 1]),
                      Transform(self.m_v_hxc_bar[1][0][0].copy(), dot_list[0][i]),
                      Transform(self.m_v_hxc_bar[1][0][1].copy(), dot_list[1][i]),
                      text_density_sq_num.animate.set_value(sq_diff_val)
                      )
            self.old_density = mol1.n_ks
            self.wait(1)
        self.wait(3)




class test2(Scene):
    def construct(self):
        arrow = DoubleArrow(LEFT, RIGHT)
        dot1 = Dot([0, 0, 0])
        self.add(arrow, dot1)
        def update_arrow(arrow, dot_ref=dot1):
            updated_arrow = DoubleArrow(dot_ref, arrow.end, tip_length=0.0, buff=0)
            arrow.become(updated_arrow)

        self.play(
            dot1.animate.move_to([3,3,0]),
            UpdateFromFunc(arrow, update_arrow, dot_ref=dot1))
        print(dir(arrow))


if __name__ == "__main__":
    from subprocess import call
    # python -m" manim" -p -ql generate_animation.py GenerateRing
    call(["python", "-m", "manim", "-p", "-ql", "generate_animation.py", "GenerateRing"])
    # call(["python", "-m", "manim", "-p", "-ql", "generate_animation.py", "test2"])
