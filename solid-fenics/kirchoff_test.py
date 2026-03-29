from dolfin import *
import numpy as np
import os

from fenics_shells import e as shell_e
from fenics_shells import gamma as rm_gamma
from fenics_shells import kirchhoff_love_theta
from fenics_shells import k as shell_k
from fenics_shells import psi_M
from fenics_shells import psi_N
from fenics_shells import psi_T

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


span = 1.0
root_chord = 0.12
tip_chord = 0.12
thickness_ratio = 0.12
leading_edge_sweep = 0.0

nx, ny = 24, 120
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, span), nx, ny)


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


# Map the unit rectangle into the physical wing planform.
coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    coords[i, 0] = x_leading_edge_at(y_val) + xi * chord_at(y_val)


def root_boundary(x, on_boundary):
    return near(x[1], 0.0) and on_boundary


U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
V = FunctionSpace(mesh, MixedElement([U_el, W_el, T_el]))

Vdisp = VectorFunctionSpace(mesh, "CG", 1, dim=3)
Vsig = TensorFunctionSpace(mesh, "DG", 0)
Vscalar = FunctionSpace(mesh, "CG", 1)

E = 6.8e10
nu = 0.35
plate_thickness = root_chord * thickness_ratio
q_load = 2.0e3  # N/m^2 downward pressure on the midsurface

du = TrialFunction(V)
v = TestFunction(V)
u = Function(V, name="PlateState")

zero_vec = Constant((0.0, 0.0))
bcs = [
    DirichletBC(V.sub(0), zero_vec, root_boundary),
    DirichletBC(V.sub(1), Constant(0.0), root_boundary),
    DirichletBC(V.sub(2), zero_vec, root_boundary),
]

I2 = Identity(2)


def split_state(x):
    return split(x)


def displacement_3d(x):
    u_mem, w, _theta = split_state(x)
    return as_vector((u_mem[0], u_mem[1], w))


def membrane_strain(u_mem):
    return shell_e(u_mem)


def curvature(theta):
    return shell_k(theta)


def k_state(x, y):
    u_x, w_x, theta_x = split_state(x)
    u_y, w_y, theta_y = split_state(y)

    eps_x = membrane_strain(u_x)
    eps_y = membrane_strain(u_y)
    kap_x = curvature(theta_x)
    kap_y = curvature(theta_y)
    gam_x = rm_gamma(theta_x, w_x)
    gam_y = rm_gamma(theta_y, w_y)

    membrane_term = (
        psi_N(
            eps_x + eps_y,
            E=Constant(E),
            nu=Constant(nu),
            t=Constant(plate_thickness),
        )
        - psi_N(eps_x, E=Constant(E), nu=Constant(nu), t=Constant(plate_thickness))
        - psi_N(eps_y, E=Constant(E), nu=Constant(nu), t=Constant(plate_thickness))
    ) * dx
    bending_term = (
        psi_M(
            kap_x + kap_y,
            E=Constant(E),
            nu=Constant(nu),
            t=Constant(plate_thickness),
        )
        - psi_M(kap_x, E=Constant(E), nu=Constant(nu), t=Constant(plate_thickness))
        - psi_M(kap_y, E=Constant(E), nu=Constant(nu), t=Constant(plate_thickness))
    ) * dx
    shear_term = (
        psi_T(
            gam_x + gam_y,
            E=Constant(E),
            nu=Constant(nu),
            t=Constant(plate_thickness),
            kappa=Constant(5.0 / 6.0),
        )
        - psi_T(
            gam_x,
            E=Constant(E),
            nu=Constant(nu),
            t=Constant(plate_thickness),
            kappa=Constant(5.0 / 6.0),
        )
        - psi_T(
            gam_y,
            E=Constant(E),
            nu=Constant(nu),
            t=Constant(plate_thickness),
            kappa=Constant(5.0 / 6.0),
        )
    ) * dx
    return membrane_term + bending_term + shear_term


def Wext(y):
    return dot(displacement_3d(y), Constant((0.0, 0.0, -q_load))) * dx


a = k_state(du, v)
L = Wext(v)

A, b = assemble_system(a, L, bcs)
solve(A, u.vector(), b, "mumps")

u_mem, w, theta = u.split(deepcopy=True)

u_vis = project(displacement_3d(u), Vdisp)
membrane_stress = project(
    (E * plate_thickness / (1.0 - nu ** 2))
    * ((1.0 - nu) * membrane_strain(u_mem) + nu * tr(membrane_strain(u_mem)) * I2),
    Vsig,
)

tip_x = x_leading_edge_at(span) + 0.75 * chord_at(span)
tip_y = span - 1.0e-8
tip_w = u(tip_x, tip_y)[2]
max_abs_disp = np.max(np.abs(u_vis.vector().get_local()))

kl_gap = kirchhoff_love_theta(w) - theta
kl_gap_norm = norm(project(sqrt(dot(kl_gap, kl_gap)), Vscalar), "l2")
elastic_energy = 0.5 * assemble(k_state(u, u))

out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)
xdmf_path = os.path.join(out_dir, "kirchoff_test_results.xdmf")
with XDMFFile(xdmf_path) as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u_vis, 0.0)
    xdmf.write(membrane_stress, 0.0)

print("Kirchhoff/Reissner-Mindlin 2D wing test completed.")
print(f"Mesh: {nx} x {ny} plate elements over span={span} m, chord={root_chord} m")
print(f"Thickness: {plate_thickness:.6e} m")
print(f"Uniform pressure load: {q_load:.3e} N/m^2")
print(f"Tip transverse displacement: {tip_w:.6e} m")
print(f"Maximum absolute displacement component: {max_abs_disp:.6e} m")
print(f"Elastic energy: {elastic_energy:.6e} J")
print(f"Kirchhoff-Love compatibility norm ||grad(w)-theta||_L2: {kl_gap_norm:.6e}")
print(f"Saved field output: {xdmf_path}")
