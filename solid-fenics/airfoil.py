from dolfin import *
from fenics import Constant, Function, AutoSubDomain, RectangleMesh, VectorFunctionSpace, interpolate, \
    TrialFunction, TestFunction, Point, Expression, DirichletBC, project, \
    Identity, inner, dx, ds, sym, grad, div, lhs, rhs, dot, File, solve, assemble_system
import numpy as np
import matplotlib.pyplot as plt
from fenicsprecice import Adapter
from enum import Enum

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary


class AirfoilFEM:
    def __init__(self,
                 L=1.0,
                 H=0.1,
                 N_chords=40,
                 Ny_per_quarter=5,
                 E=1e6,
                 nu=0.3,
                 rho=3000):

        self.L = L
        self.H = H
        self.N_chords = N_chords

        # Mesh resolution
        self.Nx = 2 * N_chords
        self.Ny = 4 * Ny_per_quarter

        self.dx = L / self.Nx
        self.dy = H / self.Ny

        # y-index for control point (3/4 height)
        self.y_cp_index = int(0.75 * self.Ny)

        # Mesh
        self.mesh = RectangleMesh(Point(0.0, 0.0),
                                  Point(L, H),
                                  self.Nx,
                                  self.Ny)

        # Function space
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)

        # Material
        self.mu = Constant(E / (2*(1+nu)))
        self.lmbda = Constant(E*nu / ((1+nu)*(1-2*nu)))

        # Boundary condition
        self.bc = DirichletBC(self.V, Constant((0.0, 0.0)), LeftBoundary())

        # Precompute force DOFs
        self.force_dofs = self._compute_force_dofs()


    def eps(self, u):
        return sym(grad(u))

    def sigma(self, u):
        return self.lmbda * div(u) * Identity(2) + 2 * self.mu * self.eps(u)

    def _compute_force_dofs(self):
        coords = self.mesh.coordinates()
        dofmap = self.V.dofmap()

        force_dofs = []

        y = self.y_cp_index * self.dy

        for i in range(self.N_chords):
            x = (2*i + 1) * self.dx  # mid-chord nodes

            for vi, c in enumerate(coords):
                if near(c[0], x) and near(c[1], y):
                    dofs = dofmap.entity_dofs(self.mesh, 0, [vi])
                    force_dofs.append(dofs)
                    break

        return force_dofs

    def step(self, cp_forces):
        """
        cp_forces[i] = (Fx, Fy) for chord i
        """

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = inner(self.sigma(u), self.eps(v)) * dx
        L = dot(Constant((0.0, 0.0)), v) * dx

        A, b = assemble_system(a, L, self.bc)

        # Apply forces
        for dofs, F in zip(self.force_dofs, cp_forces):
            b[dofs[0]] += F[0]
            b[dofs[1]] += F[1]

        uh = Function(self.V)
        solve(A, uh.vector(), b)

        return uh

if __name__ == "__main__":

    fem = AirfoilFEM(L=1.0, H=0.1, N_chords=5, Ny_per_quarter=2)

    cp_forces = [(0.0, -0.1) for _ in range(fem.N_chords)]

    u = fem.step(cp_forces)

    print("Simulation finished. Open deformation.pvd in ParaView.")

    coords = fem.mesh.coordinates()
    u_vals = u.compute_vertex_values(fem.mesh)
    ux = u_vals[0::2]
    uy = u_vals[1::2]

    scale = 50

    x_def = coords[:,0] + scale * ux
    y_def = coords[:,1] + scale * uy

    plt.figure(figsize=(10,2))
    plt.triplot(coords[:,0], coords[:,1], fem.mesh.cells(), color="gray", linewidth=0.3)
    plt.triplot(x_def, y_def, fem.mesh.cells(), color="red", linewidth=0.3)
    plt.axis("equal")
    plt.title("Deformed shape (red, scaled)")
    plt.show()