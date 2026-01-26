from dolfin import *
import numpy as np
from fenics_shells import *
import matplotlib.pyplot as plt

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["optimize"] = True


class AirfoilFEM:
    def __init__(self,
                 L=1.0,
                 H=0.1,
                 N_chords=50,
                 Ny_per_quarter=5):

        # ---- Geometry ----
        self.L = L
        self.H = H
        self.N_chords = N_chords

        self.Nx = 2 * N_chords
        self.Ny = 4 * Ny_per_quarter
        self.dx = L / self.Nx
        self.dy = H / self.Ny
        self.y_cp = 0.75 * H

        self.mesh = RectangleMesh(Point(0, 0), Point(L, H), self.Nx, self.Ny)

        # ---- Mixed shell element (NO LAGRANGE MULTIPLIER) ----
        element = MixedElement([
            VectorElement("Lagrange", triangle, 2),  # theta
            FiniteElement("Lagrange", triangle, 1),  # w
            FiniteElement("N1curl", triangle, 1)     # R_gamma
        ])

        self.Q = ProjectedFunctionSpace(self.mesh, element, num_projected_subspaces=1)
        self.QF = self.Q.full_space

        self.q_ = Function(self.QF, name="q")
        self.q  = TrialFunction(self.QF)
        self.qt = TestFunction(self.QF)

        theta_, w_, Rg_ = split(self.q_)
        theta,  w,  Rg  = split(self.q)
        theta_t, w_t, Rg_t = split(self.qt)

        E     = Constant(1e7)
        nu    = Constant(0.3)
        t     = Constant(1.0)
        rho   = Constant(1.0)
        kappa = Constant(5.0/6.0)

        mu = E / (2*(1+nu))
        lmbda = E*nu / ((1+nu)*(1-2*nu))
        kappa_theta = sym(grad(theta_))
        gamma = grad(w_) - theta_

        # ---- Energies ----
        D = (E*t**3)/(12*(1 - nu**2))
        psi_b = 0.5*D*((1-nu)*tr(kappa_theta*kappa_theta)
                       + nu*(tr(kappa_theta))**2)

        psi_s = 0.5*(E*kappa*t/(2*(1+nu))) * inner(gamma - Rg_, gamma - Rg_)

        self.Pi = (psi_b + psi_s)*dx
        self.dPi = derivative(self.Pi, self.q_, self.qt)
        self.J   = derivative(self.dPi, self.q_, self.q)

        # ---- Mass matrix (w only) ----
        self.M = assemble(rho * t * inner(w, w_t) * dx)

        self._compute_control_dofs()

    # --------------------------------------------------------

    def _compute_control_dofs(self):
        self.w_dofs = []
        coords = self.mesh.coordinates()
        dofmap = self.QF.dofmap()

        for i in range(self.N_chords):
            x = (2*i + 1) * self.dx
            for vi, c in enumerate(coords):
                if near(c[0], x, 1e-10) and near(c[1], self.y_cp, 1e-10):
                    dofs = dofmap.entity_dofs(self.mesh, 0, [vi])

                    self.w_dofs.append(dofs[2])  # w component
                    break

    def run_dynamic_generalized_alpha(self, force_history, dt, rho_inf=0.7):

        # ---- Generalized-alpha parameters ----
        alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
        alpha_f = rho_inf/(rho_inf + 1)
        gamma   = 0.5 + alpha_f - alpha_m
        beta    = 0.25*(1 + alpha_f - alpha_m)**2

        q_n     = Function(self.QF)
        qdot_n  = Function(self.QF)
        qddot_n = Function(self.QF)

        q_hist = []

        # ---- Boundary conditions (CLAMP ROOT) ----
        def root(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bc_theta = DirichletBC(self.QF.sub(0), Constant((0.0, 0.0)), root)
        bc_w     = DirichletBC(self.QF.sub(1), Constant(0.0), root)
        bcs = [bc_theta, bc_w]

        solver = PETScLUSolver("mumps")

        for n in range(len(force_history)):

            # ---- Assemble stiffness ----
            K = assemble(self.J)

            # ---- Effective stiffness ----
            a0 = (1 - alpha_m)/(beta*dt**2)
            a1 = (1 - alpha_f)

            A = K.copy()
            A *= a1
            A.axpy(a0, self.M, True)

            # ---- Internal force ----
            fint = assemble(self.dPi)

            # ---- External force ----
            fext = Vector(fint)
            fext.zero()
            for dof, f in zip(self.w_dofs, force_history[n]):
                fext[dof] += f

            # ---- Effective RHS ----
            b = fext - fint
            b += self.M * (
                a0*q_n.vector()
                + (1-alpha_m)/(beta*dt)*qdot_n.vector()
                + (1-alpha_m)*(1/(2*beta)-1)*qddot_n.vector()
            )

            for bc in bcs:
                bc.apply(A, b)

            # ---- Solve ----
            dq = Function(self.QF)
            solver.solve(A, dq.vector(), b)

            # ---- Update acceleration ----
            qddot_np1 = Function(self.QF)
            qddot_np1.vector()[:] = (
                a0*(dq.vector() - q_n.vector())
                - (1/(beta*dt))*qdot_n.vector()
                - (1/(2*beta)-1)*qddot_n.vector()
            )

            # ---- Update velocity ----
            qdot_np1 = Function(self.QF)
            qdot_np1.vector()[:] = (
                qdot_n.vector()
                + dt*((1-gamma)*qddot_n.vector() + gamma*qddot_np1.vector())
            )

            # ---- Update displacement (CRITICAL FIX) ----
            q_n.vector().axpy(1.0, dq.vector())
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
            [0.001*np.sin(0.1*n) for _ in range(fem.N_chords)]
        )

    q_hist = fem.run_dynamic_generalized_alpha(force_history, dt=0.01)

    theta, w, Rg = q_hist[-1].split()
    coords = fem.mesh.coordinates()
    w_vals = w.compute_vertex_values(fem.mesh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(coords[:,0], coords[:,1], w_vals,
                    triangles=fem.mesh.cells())
    plt.show()
