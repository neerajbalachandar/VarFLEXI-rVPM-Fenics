from dolfin import *
import numpy as np
from fenics_shells import *
import matplotlib.pyplot as plt


class AirfoilFEM:
    def __init__(self,
                 L=1.0,
                 H=0.1,
                 N_chords=50,
                 Ny_per_quarter=5,
                 rho_val=1200.0):

        # ---------------- Geometry ----------------
        self.L = L
        self.H = H
        self.N_chords = N_chords

        self.Nx = 2 * N_chords
        self.Ny = 4 * Ny_per_quarter
        self.dx = L / self.Nx
        self.dy = H / self.Ny
        self.y_cp = 0.75 * H

        self.mesh = RectangleMesh(Point(0, 0), Point(L, H),
                                  self.Nx, self.Ny)

        # ---------------- Function space ----------------
        element = MixedElement([
            VectorElement("Lagrange", triangle, 2),  # theta
            FiniteElement("Lagrange", triangle, 1),  # w
            FiniteElement("N1curl", triangle, 1),    # R_gamma
            FiniteElement("N1curl", triangle, 1)     # p
        ])

        self.Q = ProjectedFunctionSpace(self.mesh, element,
                                        num_projected_subspaces=2)
        self.QF = self.Q.full_space

        self.q_ = Function(self.QF)
        self.q = TrialFunction(self.QF)
        self.qt = TestFunction(self.QF)

        # Time states
        self.q_n = Function(self.QF)
        self.qdot_n = Function(self.QF)
        self.qddot_n = Function(self.QF)

        # ---------------- Material ----------------
        E = Constant(1.0e7)
        nu = Constant(0.3)
        kappa = Constant(5.0/6.0)
        self.t = Constant(1.0)
        self.rho = Constant(rho_val)

        theta, w, Rg, p = split(self.q_)
        theta_t, w_t, _, _ = split(self.qt)

        k = sym(grad(theta))
        gamma = grad(w) - theta

        D = (E*self.t**3)/(12*(1-nu**2))
        psi_b = 0.5*D*((1-nu)*tr(k*k) + nu*(tr(k))**2)
        psi_s = (E*kappa*self.t/(4*(1+nu)))*inner(Rg, Rg)

        # Constraint (shear locking control)
        dSp = Measure("dS", metadata={"quadrature_degree": 1})
        dsp = Measure("ds", metadata={"quadrature_degree": 1})
        n = FacetNormal(self.mesh)
        tvec = as_vector((-n[1], n[0]))

        inner_e = lambda x, y: (
            inner(x, tvec)*inner(y, tvec))("+")*dSp + \
            (inner(x, tvec)*inner(y, tvec))("-")*dSp + \
            inner(x, tvec)*inner(y, tvec)*dsp

        Pi_R = inner_e(gamma - Rg, p)

        self.Pi = psi_b*dx + psi_s*dx + Pi_R
        self.dPi = derivative(self.Pi, self.q_, self.qt)

        self._compute_control_dofs()

    # -------------------------------------------------
    def _compute_control_dofs(self):
        self.w_dofs = []
        coords = self.mesh.coordinates()
        dofmap = self.Q.dofmap()

        for i in range(self.N_chords):
            x = (2*i + 1)*self.dx
            for vi, c in enumerate(coords):
                if near(c[0], x, 1e-10) and near(c[1], self.y_cp, 1e-10):
                    dofs = dofmap.entity_dofs(self.mesh, 0, [vi])
                    self.w_dofs.append(dofs[1])
                    break

    # -------------------------------------------------
    def step_dynamic(self, cp_forces, dt):

        beta = Constant(0.25)
        gamma = Constant(0.5)
        dtc = Constant(dt)

        theta, w, _, _ = split(self.q_)
        theta_t, w_t, _, _ = split(self.qt)

        # Newmark effective acceleration
        theta, w, _, _ = split(self.q_)
        theta_n, w_n, _, _ = split(self.q_n)
        theta_dot_n, w_dot_n, _, _ = split(self.qdot_n)
        theta_ddot_n, w_ddot_n, _, _ = split(self.qddot_n)
        
        a_theta = (theta - theta_n - dtc*theta_dot_n)/(beta*dtc**2) \
                  - (1 - 2*beta)/(2*beta)*theta_ddot_n
        
        a_w = (w - w_n - dtc*w_dot_n)/(beta*dtc**2) \
      - (1 - 2*beta)/(2*beta)*w_ddot_n


        # Mass contribution
        W_mass = (
            self.rho*self.t**3 * inner(theta_t, a_theta)*dx +
            self.rho*self.t * w_t*a_w*dx
        )

        R = self.dPi + W_mass
        J = derivative(R, self.q_, self.q)

        A, b = assemble(self.Q, J, -R)

        # Clamped leading edge
        def left(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bcs = [
            DirichletBC(self.Q.sub(0), Constant((0.0, 0.0)), left),  # theta
            DirichletBC(self.Q.sub(1), Constant(0.0), left)         # w
        ]

        for bc in bcs:
            bc.apply(A, b)



        # Aerodynamic forces
        for dof, f in zip(self.w_dofs, cp_forces):
            b[dof] += -f

        solver = PETScLUSolver("mumps")
        solver.solve(A, self.q_.vector(), b)

        # Update kinematics
        qddot_new = (self.q_ - self.q_n
                     - dtc*self.qdot_n)/(beta*dtc**2) \
                    - (1 - 2*beta)/(2*beta)*self.qddot_n

        qdot_new = self.qdot_n + dtc*(
            (1-gamma)*self.qddot_n + gamma*qddot_new)

        self.q_n.assign(self.q_)
        self.qdot_n.assign(qdot_new)
        self.qddot_n.assign(qddot_new)

        return self.q_


# =====================================================
# Example run (like your previous static example)
# =====================================================
if __name__ == "__main__":

    fem = AirfoilFEM()
    dt = 1e-3
    T = 0.05
    t = 0.0

    tip_disp = []

    while t < T:
        # Example: sinusoidal aerodynamic loading
        cp_forces = [
            10.0*np.sin(2*np.pi*10*t)
            for _ in range(fem.N_chords)
        ]

        q = fem.step_dynamic(cp_forces, dt)
        _, w, _, _ = q.split()

        tip_disp.append(w.vector().max())
        t += dt

    plt.plot(tip_disp)
    plt.xlabel("Time step")
    plt.ylabel("Max deflection")
    plt.title("Dynamic airfoil response")
    plt.grid()
    plt.show()
