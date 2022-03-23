from manim import *
import numpy as np

Ns = 4
v_ext = np.array([2, 4, 5, 6])
v_hxc = np.array([0, 0, 1, 1])
t = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
eig_vec = np.random.random((Ns, Ns))
eig_val = np.random.random(Ns)
gamma = np.zeros((Ns, Ns))
pmv = 3
Ne = 4

param_dict = {0: {'v': -pmv, 'U': 5}, 1: {'v': pmv, 'U': 5}, 2: {'v': pmv, 'U': 5}, 3: {'v': -pmv, 'U': 5},
              4: {'v': pmv, 'U': 5}, 5: {'v': pmv, 'U': 5}}
t_dict = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}
equivalent_sites = [[0, 2], [1, 3]]


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

        m_v_ext_vec = Matrix(v_ext.reshape(Ns, -1))
        m_v_hxc_vec = Matrix(v_hxc.reshape(Ns, -1))

        scene3 = VGroup(VGroup(text_v_ext, m_v_ext_vec).arrange(DOWN, buff=1),
                        VGroup(text_v_hxc, m_v_hxc_vec).arrange(DOWN, buff=1)).arrange(RIGHT, buff=8)
        self.play(FadeIn(scene3))
        self.wait()

        m_v_ext = Matrix(np.diag(v_ext))

        m_v_ext.move_to(scene3[0][1])

        m_v_hxc = Matrix(np.diag(v_hxc))
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
        m_v_s = Matrix(np.diag(v_ext) + np.diag(v_hxc)).move_to(scene4[3]).scale(0.65)
        self.play(
            ReplacementTransform(text_v_ext, text_v_s),
            ReplacementTransform(m_v_hxc, m_v_s),
            FadeOut(m_v_ext),
            FadeOut(plus2),
            FadeOut(text_v_hxc)
        )

        text_h_ks_top = MathTex(r"H^{KS}").move_to(scene4[1].get_center() + UP * 2).scale(0.65)
        m_h_ks = Matrix(np.diag(v_ext) + np.diag(v_hxc) + t).move_to(scene4[1]).scale(0.65)
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
        self.add(NumberPlane())
        theta = np.pi / 4
        r = 1.5
        points = []
        lines = []
        sites_param = []
        edge_param = []
        for index1, i in enumerate(np.linspace(theta, np.pi * 2 + theta, Ns, False)):
            points.append(Circle().move_to([r * np.cos(i), r * np.sin(i), 0]).set_color(WHITE).scale(0.3))
            sites_param.append(
                Text(f'U={param_dict[index1]["U"]}, v_ext={param_dict[index1]["v"]}', font_size=23).move_to(
                    [1.5 * r * np.cos(i), 1.5 * r * np.sin(i), 0]))
            if index1 != Ns - 1:
                edge_param.append(
                    Text(f't={t_dict[(index1, index1 + 1)]}', font_size=23).move_to(
                        [1.5 * r * np.cos(i + np.pi / Ns), 1.5 * r * np.sin(i + np.pi / Ns), 0]))

            if index1 % 2 == 0:
                points[-1].set_fill(WHITE, opacity=1)
            else:
                points[-1].set_fill(BLACK, opacity=1)
        for i in range(len(points) - 1):
            lines.append(Line(points[i].get_center(), points[i + 1].get_center()).set_color(WHITE))
        edge_param.append(
            Text(f't={t_dict[(len(points) - 1, 0)]}', font_size=23).move_to(
                [1.25 * r * np.cos(theta - np.pi / Ns), 1.25 * r * np.sin(theta - np.pi / Ns), 0]))
        lines.append(Line(points[0].get_center(), points[-1].get_center()).set_color(WHITE))
        molecule = Group(*lines, *points)
        sites_param = Group(*sites_param)
        edge_param = Group(*edge_param)
        text_electrons = Text(f'{Ne}e', font_size=50)


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
        m_v_ext_vec = Matrix(v_hxc_horizontal).scale(0.75)

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
        v_ext = np.array([2, 4, 5, 6])
        v_hxc = np.array([0, 0, 1, 1])
        m_v_ext_main = Matrix(v_ext.reshape(Ns, -1)).move_to([-4, -2, 0])
        m_v_hxc_main = m_v_ext_vec.copy()
        self.play(
            FadeIn(m_v_ext_main),
            Transform(m_v_hxc_main, Matrix(v_hxc.reshape(Ns, -1)).move_to([4, -2, 0]))
        )
        self.wait()
        self.play(
            transform_into(m_v_ext_main, Matrix(np.diag(v_ext))),
            transform_into(m_v_hxc_main, Matrix(np.diag(v_hxc))),
            transform_into(text_v_ext, MathTex(r"{V}^{ext}")),
            transform_into(text_v_hxc, MathTex(r"{V}^{Hxc}"))
        )
        self.wait()

        t = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        m_t = Matrix(t).move_to([-3, -1, 0]).scale(0.65)
        text_t = MathTex(r"t").move_to([-3, 1, 0])
        plus1 = MathTex(r"+").move_to([-1,-1,0])
        plus2 = MathTex(r"+").move_to([3,-1,0])
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
            Transform(m_v_hxc_main, Matrix(np.diag(v_hxc+v_ext)).move_to(m_v_ext_main).scale(0.65))
        )
        self.wait()
        self.play(
            Transform(text_v_hxc, MathTex(r"{H}^{KS}").move_to(text_t)),
            FadeOut(text_t, m_t, plus1),
            Transform(m_v_hxc_main, Matrix(np.diag(v_hxc+v_ext)+t).move_to(m_t).scale(0.65))
        )
        ks_h_matrix = m_v_hxc_main
        self.wait()
        formula1 = MathTex(r'\hat H^{KS}', r'\ket{\Phi_0}', '=', r'\varepsilon^{KS}_0', r'\ket {\Phi_0}').move_to([3, -1, 0])
        self.play(FadeIn(formula1))
        self.wait(1)
        self.play(FadeOut(ks_hamiltonian3, text_v_hxc,formula1[0]), ks_h_matrix.animate.shift(2*LEFT))
        question_vector = Matrix(np.array(['?']*Ns).reshape(Ns, -1)).scale(0.65).next_to(ks_h_matrix, RIGHT, buff=0.5)
        self.play(ReplacementTransform(formula1[1], question_vector, buff=0.5))
        self.play(formula1[2].animate.next_to(question_vector, RIGHT, buff=0.5))
        self.play(formula1[3].animate.next_to(formula1[2], RIGHT, buff=0.5))
        question_vector2 = question_vector.copy().next_to(formula1[3], RIGHT, buff=0.5)
        self.play(ReplacementTransform(formula1[4], question_vector2))
        text1 = Text('Eigenvector problem', font_size=25).move_to([-1, -3, 0])
        temp1 = '<'
        text3 = MathTex(r'\ket{\Phi_'+ str(0) + '}' + temp1 + r'N_e/2='+str(Ne//2)).move_to([-2, 1, 0])
        global gamma
        t_gamma = DecimalMatrix(gamma,element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to([4, -2.5, 0])
        text4 = Text('1RDM').move_to([4, -1, 0])
        self.play(FadeIn(text1, text3, t_gamma, text4))
        self.wait(0.5)
        for id1 in range(Ns):
            if id1 < Ne // 2:
                temp1 = '<'
            else:
                temp1 = r'\nless'
            self.play(transform_into(question_vector, DecimalMatrix(eig_vec[:, id1].reshape(Ns, -1),element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)),
                      transform_into(question_vector2, DecimalMatrix(eig_vec[:, id1].reshape(Ns, -1),element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)),
                      transform_into(formula1[3], Text(f"{eig_val[id1]:5.2f}").scale(0.65)),
                      transform_into(text3, MathTex(r'\ket{\Phi_'+ str(id1) + '}' + temp1 + r' N_e/2='+str(Ne//2)))
            )
            if id1 < Ne // 2:
                wf_mult = question_vector.copy()
                wf_mult_T = question_vector.copy()
                self.play(wf_mult.animate.move_to([2.5, 0.5, 0]))
                self.play(
                    Transform(wf_mult_T, DecimalMatrix(eig_vec[:, id1][np.newaxis],element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to([5, 0.5, 0]))
                )
                vec_i = eig_vec[:, id1][np.newaxis]
                gamma += vec_i.T @ vec_i
                self.play(transform_into(t_gamma,DecimalMatrix(gamma,element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)),
                          FadeOut(wf_mult, wf_mult_T, SHIFT=DOWN))
            self.wait()
        self.wait(3)
        self.play(FadeOut(text1, text3, question_vector, question_vector2, formula1[3], title1, ks_h_matrix))





class test(Scene):
    def construct(self):
        self.add(NumberPlane())
        theta = np.pi / 4
        r = 1.5
        points = []
        lines = []
        sites_param = []
        edge_param = []
        for index1, i in enumerate(np.linspace(theta, np.pi * 2 + theta, Ns, False)):
            points.append(Circle().move_to([r * np.cos(i), r * np.sin(i), 0]).set_color(WHITE).scale(0.3))
            sites_param.append(
                Text(f'U={param_dict[index1]["U"]}, v_ext={param_dict[index1]["v"]}', font_size=23).move_to(
                    [1.5 * r * np.cos(i), 1.5 * r * np.sin(i), 0]))
            if index1 != Ns - 1:
                edge_param.append(
                    Text(f't={t_dict[(index1, index1 + 1)]}', font_size=23).move_to(
                        [1.5 * r * np.cos(i + np.pi / Ns), 1.5 * r * np.sin(i + np.pi / Ns), 0]))

            if index1 % 2 == 0:
                points[-1].set_fill(WHITE, opacity=1)
            else:
                points[-1].set_fill(BLACK, opacity=1)
        for i in range(len(points) - 1):
            lines.append(Line(points[i].get_center(), points[i + 1].get_center()).set_color(WHITE))
        edge_param.append(
            Text(f't={t_dict[(len(points) - 1, 0)]}', font_size=23).move_to(
                [1.25 * r * np.cos(theta - np.pi / Ns), 1.25 * r * np.sin(theta - np.pi / Ns), 0]))
        lines.append(Line(points[0].get_center(), points[-1].get_center()).set_color(WHITE))
        molecule = Group(*lines, *points)
        text_electrons = Text(f'{Ne}e', font_size=50)
        molecule = molecule.scale(0.6).move_to([6, 3, 0])
        text_electrons = text_electrons.move_to([6, 3, 0])

        v_hxc_horizontal = np.array([[0.0, 0.00, 0.01, 0]])
        m_v_ext_vec = Matrix(v_hxc_horizontal).scale(0.75)
        vec1 = MathTex(r"\vec v^{Hxc} = ")
        v_hxc_horizontal = Group(vec1, m_v_ext_vec).arrange(RIGHT, buff=0.5).move_to([0, -1, 0]).move_to([2.9, 3, 0]).scale(0.75)
        y0 = 2
        x0 = 0.5
        separating_line1 = Line([7, y0, 0], [-7, y0, 0])
        separating_line2 = Line([x0, y0, 0], [x0, 4, 0])
        title1 = Text(r"Kohn-Sham system", font_size=45, color=YELLOW).move_to([-3.25, 3, 0])
        t_h_ks = DecimalMatrix(np.diag(v_hxc + v_ext) + t, element_to_mobject_config={"num_decimal_places": 2}).move_to([5, 0.5, 0]).scale(0.65)
        # global gamma
        gamma = np.random.random((Ns, Ns))
        t_gamma = DecimalMatrix(gamma,element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to([5, -2.5, 0])
        text_gamma = Tex(r'$\gamma=$').next_to(t_gamma, LEFT)
        text_h_ks = Tex(r'$H^{KS}=$').next_to(t_h_ks, LEFT)
        text1 = Text(' ').move_to([-3, 1.5, 0])
        self.add(separating_line1, separating_line2, title1, v_hxc_horizontal, molecule, text_electrons, t_h_ks, t_gamma,
                 text_h_ks, text_gamma)
        # CASCI
        for impurity_id in range(2):
            self.play(Transform(text1, MarkupText(f'Impurity number:<span fgcolor="{YELLOW}">{impurity_id}</span>', font_size=30).move_to([-4.5, 1.5, 0])))
            squared_sites = []
            for i in equivalent_sites[impurity_id]:
                squared_sites.append(SurroundingRectangle(mobject=molecule[-Ns + i],color=YELLOW, buff=0.15))
                self.play(Create(squared_sites[-1]))
            gamma2 = gamma.copy()
            if 0 in equivalent_sites[impurity_id]:
                text2 = Text(f"Impurity is in correct place for Householder transformation.\n"
                             f"We don't need to cahnge indices", font_size=20).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait(1)
            else:
                site_id_old = int(equivalent_sites[impurity_id][0])
                print(site_id_old, gamma2)
                text2 = Text(f"Impurity is not in correct place for Householder transformation.\n"
                             f"We need to cahnge indices {site_id_old} and 0", font_size=20).move_to([-3, 0.5, 0])
                self.play(FadeIn(text2))
                self.wait()
                h_ks2 = np.diag(v_hxc + v_ext) + t

                gamma2[:, [0, site_id_old]] = gamma2[:, [site_id_old, 0]]
                h_ks2[:, [0, site_id_old]] = h_ks2[:, [site_id_old, 0]]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)))
                gamma2[[0, site_id_old], :] = gamma2[[site_id_old, 0], :]
                self.play(transform_into(t_gamma, DecimalMatrix(gamma2, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)),
                          transform_into(t_h_ks, DecimalMatrix(h_ks2, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65)))
            self.play(FadeOut(text2))
            arrow = Arrow([2, -2.5, 0], [0.5, -2.5, 0])
            self.play(Create(arrow))

            P = np.random.random((Ns, Ns))
            t_P = DecimalMatrix(P, element_to_mobject_config={"num_decimal_places": 2}).scale(0.65).move_to([-1.5, -2.5, 0])
            text_P = Tex('P=').next_to(t_P, LEFT)
            self.play(Create(t_P), FadeIn(text_P))
            arrow2 = Arrow([-1.5, -1, 0], [-1.5, 0, 0])
            eq1 = MathTex(r'\tilde h = P \cdot H^{KS} \cdot P').move_to([-1.5, 0.5, 0])

            self.play(Create(arrow2), FadeIn(eq1))
            arrow3 = Arrow([0.5, 0.5, 0], [1.7, 0.5, 0])

            self.play(Create(arrow3), transform_into(text_h_ks, MathTex(r'\tilde h=')))
            h_tilde = np.random.random((Ns, Ns)).astype("|U3")
            self.play(transform_into(t_h_ks, Matrix(h_tilde).scale(0.65)))


            self.play(FadeOut(*squared_sites))
            self.wait(1)

if __name__ == "__main__":
    from subprocess import call
    call(["python", "-m", "manim", "-p", "-ql", "generate_animation.py", "test"])