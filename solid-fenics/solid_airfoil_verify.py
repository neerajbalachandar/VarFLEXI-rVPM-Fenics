from dolfin import *
from fenics import Constant, Function, AutoSubDomain, RectangleMesh, VectorFunctionSpace, interpolate, \
    TrialFunction, TestFunction, Point, Expression, DirichletBC, project, \
    Identity, inner, dx, ds, sym, grad, div, lhs, rhs, dot, File, solve, assemble_system
import numpy as np
import matplotlib.pyplot as plt
from fenicsprecice import Adapter
from enum import Enum

from fenics_shells import *
#from fenics_shells.functions.functionspace import ProjectedFunctionSpace
#from fenics_shells.reissner_mindlin import forms,function_spaces

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class AirfoilFEM:
    def __init__(self):
        #for now fill init and make a proper initiation of mesh with proper bcs
        #get all these as inputs to __init__
        chord_length=0.1
        dim=2  
        H=0.1
        W=1
        rho=3000
        E=1
        nu=0.3
        n_x=100
        n_y=20
        mu = Constant(E / (2.0 * (1.0 + nu)))
        lambda_ = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

        #idk how this works ill get every chord length and we need to go to middle of every chord (the node closest) and in y direction 3/4th the dist and apply thyhe forces on it
        # we need to apply forces to required nodes only (the control points, other points neednt have ny forcing)
        #you may use your function for forces, distributed at every chord at 3/4th the chord length
        #get the chord length from flow-unsteady part
        #currently the code is for a plate (rectangular), which is uniformly loaded
        self.chord_length=chord_length#is this needed even, since W and n_x are given i think chord length can be found
        self.H=H
        self.W=W
        self.E=E
        self.nu=nu
        self.rho=rho
        self.nx=n_x
        self.ny=n_y

        mesh = RectangleMesh(Point(0.0, 0.0), Point(W, H), self.nx, self.ny)


        element = MixedElement([VectorElement("Lagrange", triangle, 2),
                                FiniteElement("Lagrange", triangle, 1),
                                FiniteElement("N1curl", triangle, 1),
                                FiniteElement("N1curl", triangle, 1)])
        
        Q = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)

        Q_F = Q.full_space
        q_ = Function(Q_F)
        theta_, w_, R_gamma_, p_ = split(q_)
        q = TrialFunction(Q_F)
        q_t = TestFunction(Q_F)
        
        E = Constant(10920.0)
        nu = Constant(0.3)
        kappa = Constant(5.0/6.0)
        t = Constant(0.001)#thickness

        #these things will most prolly come into the step function. i will write it here for now
        k = sym(grad(theta_))

        D = (E*t**3)/(12.0*(1.0 - nu**2))
        psi_b = 0.5*D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

        psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

        f = Constant(1.0)
        W_ext = inner(f*t**3, w_)*dx

        gamma = grad(w_) - theta_

        dSp = Measure('dS', metadata={'quadrature_degree': 1})
        dsp = Measure('ds', metadata={'quadrature_degree': 1})

        n = FacetNormal(mesh)
        t = as_vector((-n[1], n[0]))

        inner_e = lambda x, y: (inner(x, t)*inner(y, t))('+')*dSp + \
                            (inner(x, t)*inner(y, t))('-')*dSp + \
                            (inner(x, t)*inner(y, t))*dsp

        Pi_R = inner_e(gamma - R_gamma_, p_)

        Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext

        dPi = derivative(Pi, q_, q_t)
        J = derivative(dPi, q_, q)

        A, b = assemble(Q, J, -dPi)

        def all_boundary(x, on_boundary):
            return on_boundary

        bcs = [DirichletBC(Q, Constant((0.0, 0.0, 0.0)), all_boundary)]

        for bc in bcs:
            bc.apply(A, b)

        q_p_ = Function(Q)
        solver = PETScLUSolver("mumps")
        solver.solve(A, q_p_.vector(), b)

        reconstruct_full_space(q_, q_p_, J, -dPi)

        save_dir = "output/"
        theta_h, w_h, R_gamma_h, p_h = q_.split()
        fields = {"theta": theta_h, "w": w_h, "R_gamma": R_gamma_h, "p": p_h}
        for name, field in fields.items():
            field.rename(name, name)
            field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
            field_file.write(field)

    def eps(self,u):
        return sym(grad(u))

    def sigma(self,u, lmbda, mu):
        return lmbda * div(u) * Identity(2) + 2 * mu * eps(u)
    

    def step(self,cp_forces):
        pass
    #apply some bc at required chord mid at 3/4th height and output the deformed mesh
    #organize \all the funcrtions in init to proper defs and take in all required inputs into init


fem=AirfoilFEM()