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
        #you may use your function for forcing.
        #get the chord length from flow-unsteady part

        self.chord_length=chord_length#is this needed even, since W and n_x are given i think chord length can be found
        self.H=H
        self.W=W
        self.E=E
        self.nu=nu
        self.rho=rho
        self.nx=n_x
        self.ny=n_y

        mesh = RectangleMesh(Point(0.0, 0.0), Point(W, H), self.nx, self.ny)

        h = Constant(H/n_y)
        V = VectorFunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

    def eps(self,u):
        return sym(grad(u))

    def sigma(self,u, lmbda, mu):
        return lmbda * div(u) * Identity(2) + 2 * mu * eps(u)
    

    def step(self,cp_forces):
        pass
    #apply some bc at required chord mid at 3/4th height and output the deformed mesh
