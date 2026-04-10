import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

TMP_CACHE_DIR = os.path.join("/tmp", "varflexi-fenics-cache")
os.makedirs(TMP_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(TMP_CACHE_DIR, "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(TMP_CACHE_DIR, "xdg"))
os.environ.setdefault("DIJITSO_CACHE_DIR", os.path.join(TMP_CACHE_DIR, "dijitso"))
os.environ.setdefault("INSTANT_CACHE_DIR", os.path.join(TMP_CACHE_DIR, "instant"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from dolfin import *
#same as v12
from fenics_shells import e as shell_e
from fenics_shells import gamma as rm_gamma
from fenics_shells import kirchhoff_love_theta
from fenics_shells import k as shell_k

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


span = 1.0
root_chord = 0.12
tip_chord = 0.12
thickness_ratio = 0.12
leading_edge_sweep = 0.0

nx, ny = 8, 24
pressure_value = 2.0e3

E = 6.8e10
nu = 0.35
kappa_shear = Constant(5.0 / 6.0)

mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, span), nx, ny)
def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta
def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta
coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    coords[i, 0] = x_leading_edge_at(y_val) + xi * chord_at(y_val)
def root_boundary(x, on_boundary):
    return near(x[1], 0.0) and on_boundary


plate_thickness = root_chord * thickness_ratio
h = Constant(plate_thickness)

U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
V = FunctionSpace(mesh, MixedElement([U_el, W_el, T_el]))

Vdisp = VectorFunctionSpace(mesh, "CG", 1, dim=3)
Vsig = TensorFunctionSpace(mesh, "DG", 0)
Vscalar = FunctionSpace(mesh, "CG", 1)

dq = TrialFunction(V)
q_test = TestFunction(V)
q = Function(V, name="PlateState")

zero_vec = Constant((0.0, 0.0))
bcs = [
    DirichletBC(V.sub(0), zero_vec, root_boundary),
    DirichletBC(V.sub(1), Constant(0.0), root_boundary),
    DirichletBC(V.sub(2), zero_vec, root_boundary),
]

I2 = Identity(2)


def split_state(x):
    try:
        return split(x)
    except Exception:
        u_mem = as_vector((x[0], x[1]))
        w = x[2]
        theta = as_vector((x[3], x[4]))
        return u_mem, w, theta


def displacement_3d(x):
    u_mem, w, _theta = split_state(x)
    return as_vector((u_mem[0], u_mem[1], w))


def membrane_strain(u_mem):
    return shell_e(u_mem)


def curvature(theta):
    return shell_k(theta)


def membrane_stress(u_mem):
    eps = membrane_strain(u_mem)
    coeff = E * plate_thickness / (1.0 - nu ** 2)
    return coeff * ((1.0 - nu) * eps + nu * tr(eps) * I2)


def bending_moment(theta):
    kap = curvature(theta)
    coeff = E * plate_thickness ** 3 / (12.0 * (1.0 - nu ** 2))
    return coeff * ((1.0 - nu) * kap + nu * tr(kap) * I2)


def k_state(x, y):
    u_x, w_x, theta_x = split_state(x)
    u_y, w_y, theta_y = split_state(y)

    eps_y = membrane_strain(u_y)
    kap_y = curvature(theta_y)
    gam_x = rm_gamma(theta_x, w_x)
    gam_y = rm_gamma(theta_y, w_y)

    G_shear = Constant(E / (2.0 * (1.0 + nu)))
    K_shear = kappa_shear * G_shear * h

    return (
        inner(membrane_stress(u_x), eps_y) * dx
        + inner(bending_moment(theta_x), kap_y) * dx
        + K_shear * inner(gam_x, gam_y) * dx
    )


def Wext(y):
    return dot(displacement_3d(y), Constant((0.0, 0.0, -pressure_value))) * dx


def local_project(v, Vout, out=None):
    dv = TrialFunction(Vout)
    v_ = TestFunction(Vout)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if out is None:
        out = Function(Vout)
    solver.solve_local_rhs(out)
    return out


A, b = assemble_system(k_state(dq, q_test), Wext(q_test), bcs)
try:
    solve(A, q.vector(), b, "mumps")
except RuntimeError:
    solve(A, q.vector(), b, "default")

u_mem, w_fun, theta_fun = q.split(deepcopy=True)

u_vis = project(displacement_3d(q), Vdisp)
u_vis.rename("Displacement", "Displacement")

sig = Function(Vsig, name="MembraneStress")
local_project(membrane_stress(u_mem), Vsig, sig)

w_cg1 = project(w_fun, Vscalar)
w_cg1.rename("TransverseDisplacement", "TransverseDisplacement")

theta_mag = project(sqrt(dot(theta_fun, theta_fun)), Vscalar)
theta_mag.rename("RotationMagnitude", "RotationMagnitude")

kl_gap_vec = kirchhoff_love_theta(w_fun) - theta_fun
kl_gap_mag = project(sqrt(dot(kl_gap_vec, kl_gap_vec)), Vscalar)
kl_gap_mag.rename("KLGapMagnitude", "KLGapMagnitude")

tip_x = x_leading_edge_at(span) + 0.75 * chord_at(span)
tip_y = span - 1.0e-8
try:
    tip_w = q(tip_x, tip_y)[2]
except RuntimeError:
    tip_w = q(tip_x, span - 1.0e-4)[2]

max_disp = float(np.max(np.abs(u_vis.vector().get_local())))
elastic_energy = 0.5 * assemble(k_state(q, q))
kl_gap_norm = norm(kl_gap_mag, "l2")

out_dir = os.path.join(SCRIPT_DIR, "..", "results", "reissner_mindlin_v12_static")
os.makedirs(out_dir, exist_ok=True)

xdmf_path = os.path.join(out_dir, "reissner_mindlin_static_fields.xdmf")
with XDMFFile(xdmf_path) as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["rewrite_function_mesh"] = False
    xdmf.write(u_vis, 0.0)
    xdmf.write(sig, 0.0)
    xdmf.write(w_cg1, 0.0)
    xdmf.write(theta_mag, 0.0)
    xdmf.write(kl_gap_mag, 0.0)

mesh_coords = mesh.coordinates()
cells = mesh.cells()
triang = mtri.Triangulation(mesh_coords[:, 0], mesh_coords[:, 1], cells)
w_vertex = w_cg1.compute_vertex_values(mesh)

plt.figure(figsize=(7.5, 3.5))
plt.tricontourf(triang, w_vertex, levels=24, cmap="viridis")
plt.triplot(triang, color="white", linewidth=0.2, alpha=0.25)
plt.colorbar(label="w [m]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Reissner-Mindlin v12 static test: transverse displacement")
png_path = os.path.join(out_dir, "transverse_displacement.png")
plt.tight_layout()
plt.savefig(png_path, dpi=180)
plt.close()

summary_path = os.path.join(out_dir, "summary.txt")
with open(summary_path, "w", encoding="ascii") as f:
    f.write("Reissner-Mindlin v12 static verification\n")
    f.write(f"mesh={nx}x{ny}\n")
    f.write(f"pressure={pressure_value:.6e} Pa\n")
    f.write(f"thickness={plate_thickness:.6e} m\n")
    f.write(f"tip_w={tip_w:.6e} m\n")
    f.write(f"max_abs_displacement={max_disp:.6e} m\n")
    f.write(f"elastic_energy={elastic_energy:.6e} J\n")
    f.write(f"kl_gap_l2={kl_gap_norm:.6e}\n")
    f.write(f"xdmf={xdmf_path}\n")
    f.write(f"png={png_path}\n")

print("Reissner-Mindlin v12 static verification completed.")
print(f"Mesh: {nx} x {ny}")
print(f"Uniform pressure: {pressure_value:.3e} N/m^2")
print(f"Tip transverse displacement: {tip_w:.6e} m")
print(f"Maximum absolute displacement component: {max_disp:.6e} m")
print(f"Elastic energy: {elastic_energy:.6e} J")
print(f"Kirchhoff-Love compatibility norm ||grad(w)-theta||_L2: {kl_gap_norm:.6e}")
print(f"Saved XDMF fields: {xdmf_path}")
print(f"Saved PNG: {png_path}")
print(f"Saved summary: {summary_path}")
