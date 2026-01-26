from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fenics_shells import *

# ----------------- FEniCS options -----------------
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["optimize"] = True


class AirfoilFEM:
    def __init__(self,
                 L=1.0,
                 H=0.1,
                 N_chords=40,
                 Ny_per_quarter=6):

        # ---- Geometry ----
        self.L = L
        self.H = H

        self.Nx = 2 * N_chords
        self.Ny = 4 * Ny_per_quarter

        self.dx = L / self.Nx
        self.dy = H / self.Ny
        self.y_mid = 0.5 * H

        self.mesh = RectangleMesh(Point(0, 0), Point(L, H),
                                  self.Nx, self.Ny)

        # ---- Mixed shell element ----
        element = MixedElement([
            VectorElement("Lagrange", triangle, 2),  # rotations θ
            FiniteElement("Lagrange", triangle, 1),  # transverse w
            FiniteElement("N1curl", triangle, 1)     # shear Rγ
        ])

        self.Q = FunctionSpace(self.mesh, element)

        self.q_ = Function(self.Q)
        self.q  = TrialFunction(self.Q)
        self.qt = TestFunction(self.Q)

        # Split trial/test once (IMPORTANT)
        self.theta_, self.w_, self.Rg_ = split(self.q_)
        self.theta,  self.w,  self.Rg  = split(self.q)
        self.theta_t, self.w_t, self.Rg_t = split(self.qt)

        # ---- Material ----
        E     = Constant(1e6)   # lowered stiffness for visibility
        nu    = Constant(0.3)
        t     = Constant(1.0)
        rho   = Constant(1.0)
        kappa = Constant(5.0/6.0)

        # ---- Kinematics ----
        kappa_theta = sym(grad(self.theta_))
        gamma = grad(self.w_) - self.theta_

        # ---- Energies ----
        D = (E*t**3)/(12*(1 - nu**2))

        psi_b = 0.5*D*((1-nu)*tr(kappa_theta*kappa_theta)
                       + nu*(tr(kappa_theta))**2)

        psi_s = 0.5*(E*kappa*t/(2*(1+nu))) * inner(gamma - self.Rg_,
                                                   gamma - self.Rg_)

        self.Pi  = (psi_b + psi_s)*dx
        self.dPi = derivative(self.Pi, self.q_, self.qt)
        self.J   = derivative(self.dPi, self.q_, self.q)

        # ---- Mass matrix ----
        self.M = assemble(
            rho*t*inner(self.w, self.w_t)*dx
          + rho*t**3/12*inner(self.theta, self.theta_t)*dx
        )

    # --------------------------------------------------------

    def run(self, force_history, dt, rho_inf=0.7):

        # ---- Generalized-α parameters (COMET) ----
        alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
        alpha_f = rho_inf/(rho_inf + 1)
        gamma   = 0.5 + alpha_f - alpha_m
        beta    = 0.25*(1 + alpha_f - alpha_m)**2

        q_n     = Function(self.Q)
        qdot_n  = Function(self.Q)
        qddot_n = Function(self.Q)

        q_hist = []

        # ---- Boundary conditions (clamped root) ----
        def root(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bcs = [
            DirichletBC(self.Q.sub(0), Constant((0.0, 0.0)), root),
            DirichletBC(self.Q.sub(1), Constant(0.0), root)
        ]

        solver = PETScLUSolver("mumps")

        load = Constant(0.0)

        for n in range(len(force_history)):

            # α-level configuration
            q_af = Function(self.Q)
            q_af.vector()[:] = (1-alpha_f)*q_n.vector() + alpha_f*q_n.vector()
            self.q_.vector()[:] = q_af.vector()

            # Assemble internal force and tangent
            K = assemble(self.J)
            fint = assemble(self.dPi)

            # Effective stiffness
            a0 = (1-alpha_m)/(beta*dt**2)
            a1 = (1-alpha_f)

            A = a1*K
            A.axpy(a0, self.M, True)

            # External distributed load
            load.assign(force_history[n])
            fext = assemble(load * self.w_t * dx)

            # Effective RHS
            b = fext - fint
            b += self.M * (
                a0*q_n.vector()
              + (1-alpha_m)/(beta*dt)*qdot_n.vector()
              + (1-alpha_m)*(1/(2*beta)-1)*qddot_n.vector()
            )

            for bc in bcs:
                bc.apply(A, b)

            # Solve increment
            dq = Function(self.Q)
            solver.solve(A, dq.vector(), b)

            # Update acceleration
            qddot_np1 = Function(self.Q)
            qddot_np1.vector()[:] = (
                a0*(dq.vector() - q_n.vector())
              - (1/(beta*dt))*qdot_n.vector()
              - (1/(2*beta)-1)*qddot_n.vector()
            )

            # Update velocity
            qdot_np1 = Function(self.Q)
            qdot_np1.vector()[:] = (
                qdot_n.vector()
              + dt*((1-gamma)*qddot_n.vector() + gamma*qddot_np1.vector())
            )

            # Update displacement
            q_n.vector().axpy(1.0, dq.vector())
            qdot_n.assign(qdot_np1)
            qddot_n.assign(qddot_np1)

            q_hist.append(q_n.copy(deepcopy=True))

        return q_hist


# ===================== MAIN =====================

if __name__ == "__main__":

    fem = AirfoilFEM()

    dt = 0.01
    steps = 200

    # Strong but linear load (for visibility)
    force = [Constant(5.0*np.sin(0.15*n)) for n in range(steps)]

    q_hist = fem.run(force, dt)

    VIS_SCALE = 50.0   # visualization scaling only

    # =========================================================
    # 1️⃣ 2D DEFORMATION SNAPSHOTS (WIDE REGION)
    # =========================================================

    coords = fem.mesh.coordinates()
    cells  = fem.mesh.cells()

    for k in [0, 50, 100, 150, 199]:
        _, w, _ = q_hist[k].split()
        w_vals = np.array([w(Point(x, y)) for x, y in coords])

        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(coords[:,0],
                         coords[:,1],
                         VIS_SCALE*w_vals,
                         triangles=cells,
                         cmap="viridis")

        ax.set_title(f"Deformation at step {k}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("scaled w")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 2️⃣ SPACE–TIME DEFLECTION MAP
    # =========================================================

    xs = np.linspace(0, fem.L, 120)
    W = np.zeros((steps, len(xs)))

    for n in range(steps):
        _, w, _ = q_hist[n].split()
        for i, x in enumerate(xs):
            W[n, i] = w(Point(x, fem.y_mid))

    plt.figure(figsize=(7,4))
    plt.contourf(xs, np.arange(steps)*dt,
                 VIS_SCALE*W, 40, cmap="viridis")
    plt.colorbar(label="scaled w")
    plt.xlabel("x")
    plt.ylabel("time")
    plt.title("Space–time transverse deflection")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 3️⃣ ANIMATION (INTERPRETABLE)
    # =========================================================

    fig, ax = plt.subplots()
    ax.set_xlim(0, fem.L)
    ax.set_ylim(-2, 2)
    line, = ax.plot([], [], lw=2)

    def update(i):
        _, w, _ = q_hist[i].split()
        ys = [VIS_SCALE*w(Point(x, fem.y_mid)) for x in xs]
        line.set_data(xs, ys)
        ax.set_title(f"time = {i*dt:.2f}")
        return line,

    ani = FuncAnimation(fig, update, frames=steps, interval=40)
    plt.show()
