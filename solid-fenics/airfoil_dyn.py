from dolfin import *
import numpy as np
from fenics_shells import *
import matplotlib.pyplot as plt



class AirfoilFEM:
    def __init__(self,
                 L=1.0,
                 H=0.1,
                 N_chords=50,
                 Ny_per_quarter=5):

        self.L = L
        self.H = H
        self.N_chords = N_chords

        self.Nx = 2 * N_chords
        self.Ny = 4 * Ny_per_quarter
        self.dx = L / self.Nx
        self.dy = H / self.Ny
        self.y_cp = 0.75 * H

        self.mesh = RectangleMesh(Point(0,0), Point(L,H), self.Nx, self.Ny)

        element = MixedElement([
            VectorElement("Lagrange", triangle, 2),  # theta
            FiniteElement("Lagrange", triangle, 1),  # w
            FiniteElement("N1curl", triangle, 1),    # R_gamma
            FiniteElement("N1curl", triangle, 1)     # p
        ])

        self.Q = ProjectedFunctionSpace(self.mesh, element, num_projected_subspaces=2)
        self.QF = self.Q.full_space

        self.q_ = Function(self.QF)
        self.q = TrialFunction(self.QF)
        self.qt = TestFunction(self.QF)

        theta_, w_, R_gamma_, p_ = split(self.q_)

        # Material
        E = Constant(1e7)
        nu = Constant(0.3)
        kappa = Constant(5.0/6.0)
        t = Constant(1.0)
        rho = Constant(1.0)

        # Kinematics
        k = sym(grad(theta_))
        gamma = grad(w_) - theta_

        # Energies
        D = (E*t**3)/(12.0*(1.0 - nu**2))
        psi_b = 0.5*D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)
        psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

        dSp = Measure('dS')
        dsp = Measure('ds')
        n = FacetNormal(self.mesh)
        tvec = as_vector((-n[1], n[0]))

        inner_e = lambda x, y: (
            inner(x, tvec)*inner(y, tvec)
        )('+')*dSp + (
            inner(x, tvec)*inner(y, tvec)
        )('-')*dSp + (
            inner(x, tvec)*inner(y, tvec)
        )*dsp

        Pi_R = inner_e(gamma - R_gamma_, p_)

        self.Pi = psi_b*dx + psi_s*dx + Pi_R
        self.dPi = derivative(self.Pi, self.q_, self.qt)
        self.J = derivative(self.dPi, self.q_, self.q)

        theta_t, w_t, Rg_t, p_t = split(self.qt)
        self.M = assemble(rho * t * inner(w_, w_t) * dx)

        self._compute_control_dofs()

    def _compute_control_dofs(self):
        self.w_dofs = []
        coords = self.mesh.coordinates()
        dofmap = self.Q.dofmap()

        for i in range(self.N_chords):
            x = (2*i + 1) * self.dx
            for vi, c in enumerate(coords):
                if near(c[0], x, 1e-10) and near(c[1], self.y_cp, 1e-10):
                    dofs = dofmap.entity_dofs(self.mesh, 0, [vi])
                    self.w_dofs.append(dofs[1])
                    break

    def run_dynamic_generalized_alpha(
        self,
        force_history,
        dt,
        rho_inf=0.7,
        eta_m=0.0,
        eta_k=0.0
    ):
        """
        Generalized-alpha time integration for nonlinear Reissner-Mindlin shell
        (Airfoil problem)
        """

        # ---- Generalized-alpha parameters ----
        alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
        alpha_f = rho_inf/(rho_inf + 1)
        gamma   = 0.5 + alpha_f - alpha_m
        beta    = 0.25*(1 + alpha_f - alpha_m)**2

        num_steps = len(force_history)

        # ---- State variables ----
        q_n     = Function(self.QF, name="q")
        qdot_n  = Function(self.QF, name="qdot")
        qddot_n = Function(self.QF, name="qddot")

        q_hist = []

        # ---- Boundary condition ----
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bc = DirichletBC(self.Q, Constant((0.0, 0.0, 0.0)), left_boundary)
        solver = PETScLUSolver("mumps")

        # ---- Time stepping ----
        for n in range(num_steps):

            cp_forces = force_history[n]

            # ---- Assemble tangent stiffness ----
            K = assemble(self.J)

            # ---- Rayleigh damping matrix ----
            C = K.copy()
            C *= eta_k
            C.axpy(eta_m, self.M, True)

            # ---- Effective stiffness ----
            a0 = (1 - alpha_m)/(beta*dt**2)
            a1 = (1 - alpha_f)
            a2 = (1 - alpha_f)*gamma/(beta*dt)

            A = K.copy()
            A *= a1
            A.axpy(a0, self.M, True)
            A.axpy(a2, C, True)

            # ---- Internal force vector ----
            fint = assemble(self.dPi)

            # ---- External force vector ----
            fext = Vector(fint)
            fext.zero()

            for dof, f in zip(self.w_dofs, cp_forces):
                fext[dof] += f

            # ---- Effective RHS ----
            b = fext - fint

            b += self.M * (
                a0*q_n.vector()
                + (1 - alpha_m)/(beta*dt)*qdot_n.vector()
                + (1 - alpha_m)*(1/(2*beta) - 1)*qddot_n.vector()
            )

            b += C * (
                a2*q_n.vector()
                + (1 - alpha_f)*(gamma/beta - 1)*qdot_n.vector()
                + (1 - alpha_f)*dt*(gamma/(2*beta) - 1)*qddot_n.vector()
            )

            bc.apply(A, b)

            # ---- Solve for displacement increment ----
            dq = Function(self.QF)
            solver.solve(A, dq.vector(), b)

            # ---- Update acceleration ----
            qddot_np1 = Function(self.QF)
            qddot_np1.vector()[:] = (
                a0*(dq.vector() - q_n.vector())
                - (1/(beta*dt))*qdot_n.vector()
                - (1/(2*beta) - 1)*qddot_n.vector()
            )

            # ---- Update velocity ----
            qdot_np1 = Function(self.QF)
            qdot_np1.vector()[:] = (
                qdot_n.vector()
                + dt*((1 - gamma)*qddot_n.vector() + gamma*qddot_np1.vector())
            )

            # ---- Update displacement ----
            q_n.assign(dq)
            qdot_n.assign(qdot_np1)
            qddot_n.assign(qddot_np1)

            q_hist.append(q_n.copy(deepcopy=True))

        return q_hist



if __name__ == "__main__":

    fem = AirfoilFEM()

    num_steps = 200
    force_history = []

    for n in range(num_steps):
        force_history.append(
            [np.sin(0.1*n) for _ in range(fem.N_chords)]
        )

    q_hist = fem.run_dynamic_generalized_alpha(force_history, dt=0.01)

    theta, w, _, _ = q_hist[-1].split()
    coords = fem.mesh.coordinates()
    w_vals = w.compute_vertex_values(fem.mesh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(coords[:,0], coords[:,1], w_vals,
                    triangles=fem.mesh.cells())
    plt.show()
