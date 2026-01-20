from fenics import Constant, Function, AutoSubDomain, RectangleMesh, VectorFunctionSpace, interpolate, \
    TrialFunction, TestFunction, Point, Expression, DirichletBC, project, \
    Identity, inner, dx, ds, sym, grad, div, lhs, rhs, dot, File, solve, assemble_system
import numpy as np
import matplotlib.pyplot as plt
from fenicsprecice import Adapter
from enum import Enum



# Define strain
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)


# Define Stress tensor
def sigma(u):
    return lambda_ * div(u) * Identity(dim) + 2 * mu * epsilon(u)


# Define Mass form
def m(u, v):
    return rho * inner(u, v) * dx


# Elastic stiffness form
def k(u, v):
    return inner(sigma(u), sym(grad(v))) * dx


# External Work
def Wext(u_):
    return dot(u_, p) * ds


# Update functions

# Update acceleration
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)

    return ((u - u_old - dt_ * v_old) / beta / dt_ ** 2
            - (1 - 2 * beta_) / 2 / beta_ * a_old)


# Update velocity
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)

    return v_old + dt_ * ((1 - gamma_) * a_old + gamma_ * a)



from dolfin import *
import numpy as np
from fenics_shells import *
from mpl_toolkits.mplot3d import Axes3D
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
            VectorElement("Lagrange", triangle, 2),
            FiniteElement("Lagrange", triangle, 1),
            FiniteElement("N1curl", triangle, 1),
            FiniteElement("N1curl", triangle, 1)
        ])

        self.Q = ProjectedFunctionSpace(self.mesh, element, num_projected_subspaces=2)
        self.QF = self.Q.full_space

        self.q_ = Function(self.QF)
        self.q = TrialFunction(self.QF)
        self.qt = TestFunction(self.QF)

        theta_, w_, R_gamma_, p_ = split(self.q_)

        E = Constant(10000000.0)
        nu = Constant(0.3)
        kappa = Constant(5.0/6.0)
        t = Constant(1)

        k = sym(grad(theta_))
        gamma = grad(w_) - theta_

        D = (E*t**3)/(12.0*(1.0 - nu**2))
        psi_b = 0.5*D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

        psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

        dSp = Measure('dS', metadata={'quadrature_degree': 1})
        dsp = Measure('ds', metadata={'quadrature_degree': 1})
        n = FacetNormal(self.mesh)
        tvec = as_vector((-n[1], n[0]))

        inner_e = lambda x, y: (inner(x, tvec)*inner(y, tvec))('+')*dSp + \
                               (inner(x, tvec)*inner(y, tvec))('-')*dSp + \
                               (inner(x, tvec)*inner(y, tvec))*dsp

        Pi_R = inner_e(gamma - R_gamma_, p_)

        W_ext = Constant(0.0) * w_ * dx

        self.Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext

        self.dPi = derivative(self.Pi, self.q_, self.qt)
        self.J = derivative(self.dPi, self.q_, self.q)

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

    def step(self, cp_forces):

        A, b = assemble(self.Q, self.J, -self.dPi)

        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)


        bcs = [DirichletBC(self.Q, Constant((0.0, 0.0, 0.0)), left_boundary)]

        for bc in bcs:
            bc.apply(A, b)

        for dof, f in zip(self.w_dofs, cp_forces):
            b[dof] += -f

        q_p = Function(self.Q)
        solver = PETScLUSolver("mumps")
        solver.solve(A, q_p.vector(), b)

        reconstruct_full_space(self.q_, q_p, self.J, -self.dPi)

        save_dir = "airfoil_output/"
        theta_h, w_h, R_gamma_h, p_h = self.q_.split()
        fields = {"theta": theta_h, "w": w_h, "R_gamma": R_gamma_h, "p": p_h}
        for name, field in fields.items():
            field.rename(name, name)
            field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
            field_file.write(field)

        return self.q_

#main
if __name__ == "__main__":
    fem = AirfoilFEM()
    cp_forces = [1.0 for _ in range(fem.N_chords)]
    q = fem.step(cp_forces)

    theta, w, _, _ = q.split()
    coords = fem.mesh.coordinates()
    w_vals = w.compute_vertex_values(fem.mesh)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(coords[:,0], coords[:,1], w_vals, triangles=fem.mesh.cells())
    plt.show()
