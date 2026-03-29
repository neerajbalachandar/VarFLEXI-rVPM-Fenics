# CHANGED FROM v11/v12-solid:
# This file now models the wing as a 2D plate midsurface instead of a 3D solid.
# The formulation uses Reissner-Mindlin primary variables (membrane
# displacement, transverse displacement, rotations), which approaches the
# Kirchhoff-Love thin-plate relation grad(w) ~= theta as thickness becomes small.
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import socket
import json
import os
from scipy.spatial import cKDTree

from fenics_shells import e as shell_e
from fenics_shells import gamma as rm_gamma
from fenics_shells import kirchhoff_love_theta
from fenics_shells import k as shell_k
from fenics_shells import psi_M
from fenics_shells import psi_N
from fenics_shells import psi_T

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

T = 4.0
Nsteps = 400
dt_value = T / Nsteps
dt = Constant(dt_value)

span = 1.0
root_chord = 0.12
tip_chord = 0.12
thickness_ratio = 0.12
leading_edge_sweep = 0.0

# CHANGED:
# The structural mesh is now a 2D midsurface mesh for the wing planform.
# In the previous solid model this was a 3D BoxMesh through the thickness.
nx, ny = 24, 120
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, span), nx, ny)

n_span = 80
n_chord = 8
m_panels_comm = n_span * n_chord
eta_span_comm = np.linspace(0.0, 1.0, n_span)
eta_chord_edges = np.linspace(0.0, 1.0, n_chord + 1)
eta_chord_comm = eta_chord_edges[:-1] + 0.75 * (
    eta_chord_edges[1:] - eta_chord_edges[:-1]
)

work_conservative_mode = True
rbf_epsilon = 1.0
work_rel_tol = 1.0e-3
work_conv_window = 10


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


plate_thickness = root_chord * thickness_ratio
h = Constant(plate_thickness)

# CHANGED:
# We still start from a unit rectangle, but now only map the in-plane chordwise
# coordinate to the physical tapered/swept wing planform. There is no 3D airfoil
# thickness projection here because the structure is a plate midsurface.
coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    chord = chord_at(y_val)
    coords[i, 0] = x_leading_edge_at(y_val) + xi * chord


def left(x, on_boundary):
    return near(x[1], 0.0) and on_boundary


def right(x, on_boundary):
    return near(x[1], span) and on_boundary


# CHANGED:
# The unknowns are now:
# u_mem  -> in-plane membrane displacement (ux, uy)
# w      -> transverse displacement
# theta  -> independent plate rotations
# This mixed space replaces the old 3D vector displacement field.
U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
state_element = MixedElement([U_el, W_el, T_el])
V = FunctionSpace(mesh, state_element)

Vt = VectorFunctionSpace(mesh, "CG", 1, dim=3)
Vsig = TensorFunctionSpace(mesh, "DG", 0)

# CHANGED:
# Aerodynamic loading is still represented as a 3-component vector field so the
# socket coupling format remains compatible with the fluid side.
t_aero = Function(Vt, name="AerodynamicLoad")

E = 6.8e10
nu = 0.35
rho_s = 1600.0
rho = Constant(rho_s)
kappa_shear = Constant(5.0 / 6.0)
eta_m = Constant(0.8)
eta_k = Constant(1.0e-4)

alpha_m = Constant(0.10)
alpha_f = Constant(0.20)
gamma = Constant(0.5 + alpha_f - alpha_m)
beta = Constant((gamma + 0.5) ** 2 / 4.0)

print(
    f"Plate setup (solid): span={span} m, c_root={root_chord} m, c_tip={tip_chord} m, "
    f"h={plate_thickness:.4e} m, E={E:.3e} Pa, rho_s={rho_s} kg/m^3"
)

# CHANGED:
# Keep a dedicated TrialFunction for Jacobian linearization of the nonlinear
# mixed plate residual.
dq_trial = TrialFunction(V)
q_test = TestFunction(V)
q = Function(V, name="PlateState")
q_old = Function(V)
v_old = Function(V)
a_old = Function(V)

u_zero = Constant((0.0, 0.0))
bc_u = DirichletBC(V.sub(0), u_zero, left)
bc_w = DirichletBC(V.sub(1), Constant(0.0), left)
bc_theta = DirichletBC(V.sub(2), u_zero, left)
bcs = [bc_u, bc_w, bc_theta]

I2 = Identity(2)


## Plate kinematics and constitutive terms
## CHANGED:
## These replace the old 3D solid strain/stress definitions.

def membrane_strain(u_mem):
    # Uses fenics_shells.common.kinematics.e
    return shell_e(u_mem)


def curvature(theta):
    # Uses fenics_shells.common.kinematics.k
    return shell_k(theta)


def membrane_stress(u_mem):
    eps = membrane_strain(u_mem)
    coeff = E * plate_thickness / (1.0 - nu ** 2)
    return coeff * ((1.0 - nu) * eps + nu * tr(eps) * I2)


def bending_moment(theta):
    kap = curvature(theta)
    coeff = E * plate_thickness ** 3 / (12.0 * (1.0 - nu ** 2))
    return coeff * ((1.0 - nu) * kap + nu * tr(kap) * I2)


def split_state(x):
    # `ufl.split(x)` works for mixed Arguments/Coefficients, but fails for
    # algebraic expressions like `avg(...)` that produce a UFL `Sum`.
    # Fall back to explicit component indexing for those cases.
    try:
        return split(x)
    except Exception:
        u_mem = as_vector((x[0], x[1]))
        w = x[2]
        theta = as_vector((x[3], x[4]))
        return u_mem, w, theta


def displacement_3d(x):
    # CHANGED:
    # Rebuild a 3D-looking displacement vector from plate unknowns so the aero
    # transfer and output paths can keep using [ux, uy, uz]-style data.
    u_mem, w, _theta = split_state(x)
    return as_vector((u_mem[0], u_mem[1], w))


def m_state(x, y):
    # CHANGED:
    # Plate inertia = translational inertia of the midsurface + rotary inertia.
    u_x, w_x, theta_x = split_state(x)
    u_y, w_y, theta_y = split_state(y)
    inertia_rot = rho * h ** 3 / 12.0
    return (
        rho * h * (inner(u_x, u_y) + w_x * w_y) * dx
        + inertia_rot * inner(theta_x, theta_y) * dx
    )


def k_state(x, y):
    # CHANGED:
    # Plate stiffness now contains membrane, bending, and transverse shear parts.
    # The shear term is what makes this a Reissner-Mindlin model rather than a
    # pure Kirchhoff-Love model.
    u_x, w_x, theta_x = split_state(x)
    u_y, w_y, theta_y = split_state(y)

    eps_y = membrane_strain(u_y)
    kap_y = curvature(theta_y)
    gam_x = rm_gamma(theta_x, w_x)
    gam_y = rm_gamma(theta_y, w_y)

    # Use explicit stress-resultant virtual work instead of polarization of
    # energy terms. This avoids mixed-form arity issues during Jacobian build.
    N_x = membrane_stress(u_x)
    M_x = bending_moment(theta_x)
    G_shear = Constant(E / (2.0 * (1.0 + nu)))
    K_shear = kappa_shear * G_shear * h

    membrane_term = inner(N_x, eps_y) * dx
    bending_term = inner(M_x, kap_y) * dx
    shear_term = K_shear * inner(gam_x, gam_y) * dx
    return membrane_term + bending_term + shear_term


def c_state(x, y):
    return eta_m * m_state(x, y) + eta_k * k_state(x, y)


def Wext(y):
    v_disp = displacement_3d(y)
    return dot(v_disp, t_aero) * dx


def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u - u_old - dt_ * v_old) / beta_ / dt_ ** 2 - (1.0 - 2.0 * beta_) / (
        2.0 * beta_
    ) * a_old


def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_ * ((1.0 - gamma_) * a_old + gamma_ * a)


def update_fields(u, u_old, v_old, a_old):
    u_vec = u.vector()
    u0_vec = u_old.vector()
    v0_vec = v_old.vector()
    a0_vec = a_old.vector()

    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    v_old.vector()[:] = v_vec
    a_old.vector()[:] = a_vec
    u_old.vector()[:] = u_vec


def avg(x_old, x_new, alpha):
    return alpha * x_old + (1.0 - alpha) * x_new


# CHANGED:
# Nonlinear residual must be written in terms of the unknown state `q`
# (not TrialFunction), otherwise `lhs/rhs` extraction fails on nonlinear terms.
a_new = update_a(q, q_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, q_old, v_old, a_old, ufl=True)

res = (
    m_state(avg(a_old, a_new, alpha_m), q_test)
    + c_state(avg(v_old, v_new, alpha_f), q_test)
    + k_state(avg(q_old, q, alpha_f), q_test)
    - Wext(q_test)
)

# CHANGED:
# Build Jacobian explicitly from nonlinear residual.
jac_form = derivative(res, q, dq_trial)

# CHANGED:
# Newton settings for nonlinear mixed plate solve.
newton_abs_tol = 1.0e-6
newton_rel_tol = 1.0e-6
newton_inc_tol = 1.0e-8
newton_max_iters = 35


def solve_nonlinear_step(q_fun, jac_form, res_form, bcs, ext_force_vec=None):
    for bc in bcs:
        bc.apply(q_fun.vector())

    residual_norm0 = None
    dq_step = Function(V)

    for it in range(newton_max_iters):
        A = assemble(jac_form)
        residual_vec = assemble(res_form)
        if ext_force_vec is not None:
            residual_vec.axpy(-1.0, ext_force_vec)

        for bc in bcs:
            bc.apply(A, residual_vec, q_fun.vector())
        A.ident_zeros()

        residual_norm = residual_vec.norm("l2")
        if residual_norm0 is None:
            residual_norm0 = max(residual_norm, 1.0)
        rel_norm = residual_norm / residual_norm0

        if residual_norm <= newton_abs_tol or rel_norm <= newton_rel_tol:
            return it, residual_norm, rel_norm

        rhs = residual_vec.copy()
        rhs *= -1.0
        try:
            lin_solver = LUSolver(A, "mumps")
        except RuntimeError:
            lin_solver = LUSolver(A, "default")
        lin_solver.parameters["symmetric"] = False
        lin_solver.solve(dq_step.vector(), rhs)

        dq_norm = dq_step.vector().norm("l2")
        q_norm = max(q_fun.vector().norm("l2"), 1.0)
        inc_rel = dq_norm / q_norm

        q_fun.vector().axpy(1.0, dq_step.vector())
        for bc in bcs:
            bc.apply(q_fun.vector())

        if inc_rel <= newton_inc_tol:
            return it + 1, residual_norm, rel_norm

    raise RuntimeError(
        f"Newton solve failed after {newton_max_iters} iterations "
        f"(abs={residual_norm:.3e}, rel={rel_norm:.3e})"
    )

## Coupling utilities retained from the old file
## CHANGED:
## The communication/data-layout logic is mostly preserved, but the geometric
## queries now operate on midsurface nodes instead of outer solid-surface nodes.
panel_node_cache = {}
coupling_node_cache = {}
max_abs_force_component = 5.0e3


def as_eta_array(values, n):
    if values is None:
        return np.linspace(0.0, 1.0, n)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != n:
        return np.linspace(0.0, 1.0, n)
    if n > 1:
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.maximum.accumulate(arr)
        if arr[-1] > 0.0:
            arr = arr / arr[-1]
    return arr


def interp_profile(x_src, vals_src, x_dst):
    x_src = np.asarray(x_src, dtype=float).reshape(-1)
    vals_src = np.asarray(vals_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float).reshape(-1)
    if vals_src.ndim == 1:
        vals_src = vals_src.reshape(-1, 1)
    if len(x_src) == 0 or vals_src.shape[0] == 0:
        return np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
    if len(x_src) == 1:
        return np.repeat(vals_src[:1, :], len(x_dst), axis=0)
    out = np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
    for c in range(vals_src.shape[1]):
        out[:, c] = np.interp(x_dst, x_src, vals_src[:, c])
    return out


def resample_forces_to_shape(
    forces,
    n_span_out,
    n_chord_out,
    eta_span_in=None,
    eta_chord_in=None,
    eta_span_out=None,
    eta_chord_out=None,
):
    forces = np.asarray(forces, dtype=float).reshape(-1, 3)
    n_out = n_span_out * n_chord_out
    if len(forces) == 0:
        return np.zeros((n_out, 3), dtype=float)

    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    n_in = len(forces)
    if eta_span_in is None or eta_chord_in is None:
        if n_in == n_out:
            return forces
        if n_in == 1:
            return np.repeat(forces[:1], n_out, axis=0)
        s_in = np.linspace(0.0, 1.0, n_in)
        s_out = np.linspace(0.0, 1.0, n_out)
        return interp_profile(s_in, forces, s_out)

    eta_span_in = np.asarray(eta_span_in, dtype=float).reshape(-1)
    eta_chord_in = np.asarray(eta_chord_in, dtype=float).reshape(-1)
    n_span_in = len(eta_span_in)
    n_chord_in = len(eta_chord_in)
    if n_span_in == 0 or n_chord_in == 0:
        return np.zeros((n_out, 3), dtype=float)
    if n_in != n_span_in * n_chord_in:
        return resample_forces_to_shape(forces, n_span_out, n_chord_out)

    grid_in = forces.reshape((n_span_in, n_chord_in, 3))
    eta_span_in = as_eta_array(eta_span_in, n_span_in)
    eta_chord_in = as_eta_array(eta_chord_in, n_chord_in)

    grid_span = np.zeros((n_span_out, n_chord_in, 3), dtype=float)
    for j in range(n_chord_in):
        grid_span[:, j, :] = interp_profile(
            eta_span_in, grid_in[:, j, :], eta_span_out
        )

    grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
    for i in range(n_span_out):
        grid_out[i, :, :] = interp_profile(
            eta_chord_in, grid_span[i, :, :], eta_chord_out
        )

    return grid_out.reshape((n_out, 3))


def get_scalar_space_coords(space):
    # CHANGED:
    # Promote 2D FE coordinates to pseudo-3D [x, y, 0] coordinates so existing
    # coupling code can keep working with 3-component point arrays.
    scalar_space = space.sub(0).collapse()
    xy = scalar_space.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    xyz = np.zeros((xy.shape[0], 3), dtype=float)
    xyz[:, :2] = xy
    return xyz


def get_panel_node_ids(n_span, n_chord, eta_chord):
    # CHANGED:
    # Panel ownership is now computed on the plate midsurface instead of the
    # upper/lower airfoil skin nodes used in the 3D solid model.
    eta_chord = as_eta_array(eta_chord, n_chord)
    key = (n_span, n_chord, tuple(np.round(eta_chord, 8)))
    if key in panel_node_cache:
        return panel_node_cache[key]

    coords_xyz = get_scalar_space_coords(Vt)
    panel_node_ids = [[] for _ in range(n_span * n_chord)]

    for i_node, X in enumerate(coords_xyz):
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            continue
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1.0e-12)
        if xi < -0.02 or xi > 1.02:
            continue
        eta_s = np.clip(y_val / span, 0.0, 1.0)
        i_span = min(int(eta_s * n_span), n_span - 1)
        j_chord = int(np.argmin(np.abs(eta_chord - xi)))
        panel_idx = i_span * n_chord + j_chord
        panel_node_ids[panel_idx].append(i_node)

    for i_span in range(n_span):
        y_target = (i_span + 0.5) * span / n_span
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord):
            panel_idx = i_span * n_chord + j_chord
            if panel_node_ids[panel_idx]:
                continue
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            distances = np.sum(
                (coords_xyz[:, :2] - np.array([[x_target, y_target]])) ** 2, axis=1
            )
            panel_node_ids[panel_idx].append(int(np.argmin(distances)))

    panel_node_cache[key] = panel_node_ids
    return panel_node_ids


def update_aero_traction(t_aero, forces, n_span, n_chord, eta_chord):
    # CHANGED:
    # Panel forces are converted directly into midsurface distributed loads.
    # There is no split across top and bottom wing skins because the structure
    # is no longer represented with two physical outer surfaces.
    vec = t_aero.vector()
    vec.zero()

    dofs_x = Vt.sub(0).dofmap().dofs()
    dofs_y = Vt.sub(1).dofmap().dofs()
    dofs_z = Vt.sub(2).dofmap().dofs()

    eta_chord = as_eta_array(eta_chord, n_chord)
    if n_chord == 1:
        chord_edges = np.array([0.0, 1.0], dtype=float)
    else:
        chord_edges = np.zeros(n_chord + 1, dtype=float)
        chord_edges[1:-1] = 0.5 * (eta_chord[:-1] + eta_chord[1:])
        chord_edges[0] = max(0.0, eta_chord[0] - 0.5 * (eta_chord[1] - eta_chord[0]))
        chord_edges[-1] = min(
            1.0, eta_chord[-1] + 0.5 * (eta_chord[-1] - eta_chord[-2])
        )
        if chord_edges[-1] <= chord_edges[0]:
            chord_edges = np.linspace(0.0, 1.0, n_chord + 1)

    panel_node_ids = get_panel_node_ids(n_span, n_chord, eta_chord)
    for panel_idx, ids in enumerate(panel_node_ids):
        if not ids:
            continue
        fx, fy, fz = forces[panel_idx]
        i_span = panel_idx // n_chord
        j_chord = panel_idx % n_chord

        y_mid = (i_span + 0.5) * span / n_span
        c_mid = chord_at(y_mid)
        d_eta_c = max(chord_edges[j_chord + 1] - chord_edges[j_chord], 1.0e-8)
        d_span = span / n_span
        panel_area = max(c_mid * d_eta_c * d_span, 1.0e-10)

        tx = fx / panel_area
        ty = fy / panel_area
        tz = fz / panel_area
        scale = 1.0 / len(ids)
        for i_node in ids:
            vec[dofs_x[i_node]] += scale * tx
            vec[dofs_y[i_node]] += scale * ty
            vec[dofs_z[i_node]] += scale * tz

    vec.apply("insert")


def parse_force_payload(data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    if "n_span" in data and "n_chord" in data and "force" in data:
        n_span_in = int(data.get("n_span", 0))
        n_chord_in = int(data.get("n_chord", 0))
        force_raw = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        eta_span_in = (
            as_eta_array(data.get("eta_span"), n_span_in) if n_span_in > 0 else None
        )
        eta_chord_in = (
            as_eta_array(data.get("eta_chord"), n_chord_in) if n_chord_in > 0 else None
        )
        if n_span_in > 0 and n_chord_in > 0:
            forces = resample_forces_to_shape(
                force_raw,
                n_span_out,
                n_chord_out,
                eta_span_in=eta_span_in,
                eta_chord_in=eta_chord_in,
                eta_span_out=eta_span_out,
                eta_chord_out=eta_chord_out,
            )
            forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
            return forces, True

    forces_legacy = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
    forces = resample_forces_to_shape(
        forces_legacy,
        n_span_out,
        n_chord_out,
        eta_span_out=eta_span_out,
        eta_chord_out=eta_chord_out,
    )
    forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
    return forces, False


def get_aero_surface_node_ids():
    # CHANGED:
    # For the plate model, the aerodynamic interface is identified with the full
    # midsurface node set.
    coords_xyz = get_scalar_space_coords(Vt)
    ids = np.arange(coords_xyz.shape[0], dtype=np.int64)
    return ids, coords_xyz


def extract_coupling_node_indices(n_span, n_chord, eta_chord, coords_xyz):
    eta_chord = as_eta_array(eta_chord, n_chord)
    tree = cKDTree(coords_xyz[:, :2])
    targets = []
    for i_span in range(n_span):
        y_target = (i_span + 0.5) * span / n_span
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord):
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            targets.append([x_target, y_target])
    _, idx = tree.query(np.asarray(targets), k=1)
    return np.asarray(idx, dtype=np.int64)


def build_local_rbf_map(fluid_points, solid_points, epsilon, n_neighbors=32):
    fluid_points = np.asarray(fluid_points, dtype=float)
    solid_points = np.asarray(solid_points, dtype=float)
    eps2 = max(float(epsilon) ** 2, 1.0e-16)
    n_s = solid_points.shape[0]
    k = int(max(1, min(n_neighbors, n_s)))

    tree = cKDTree(solid_points)
    d, idx = tree.query(fluid_points, k=k)
    if k == 1:
        d = d.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    nbr_ids = idx.astype(np.int64)
    r2 = d * d
    nbr_w = np.exp(-r2 / eps2)
    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
    for bi in bad:
        nbr_w[bi, :] = 0.0
        nbr_w[bi, 0] = 1.0
    row_sum = np.maximum(row_sum, 1.0e-16)
    nbr_w /= row_sum
    return nbr_ids, nbr_w


def map_displacements_to_fluid(u_nodes, nbr_ids, nbr_w):
    n_f, k = nbr_ids.shape
    out = np.zeros((n_f, 3), dtype=float)
    for q_idx in range(k):
        out += nbr_w[:, q_idx : q_idx + 1] * u_nodes[nbr_ids[:, q_idx], :]
    return out


def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
    n_f, k = nbr_ids.shape
    out = np.zeros((n_solid_nodes, 3), dtype=float)
    for q_idx in range(k):
        contrib = nbr_w[:, q_idx : q_idx + 1] * f_fluid
        np.add.at(out, nbr_ids[:, q_idx], contrib)
    return out


def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
    _n_f, k = nbr_ids.shape
    S = np.zeros((n_solid_nodes,), dtype=float)
    for q_idx in range(k):
        np.add.at(S, nbr_ids[:, q_idx], nbr_w[:, q_idx] * A_diag)
    return np.maximum(S, 1.0e-14)


def apply_Tf_operator(Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
    FaA = Fa * A_diag[:, None]
    rhs = map_forces_to_solid(FaA, n_solid_nodes, nbr_ids, nbr_w)
    Fs_coeff = rhs / S_lumped[:, None]
    return Fs_coeff, rhs


def get_nodal_displacements(q_fun, node_ids, dofs_x, dofs_y, dofs_w):
    q_arr = q_fun.vector().get_local()
    out = np.zeros((len(node_ids), 3), dtype=float)
    out[:, 0] = q_arr[dofs_x[node_ids]]
    out[:, 1] = q_arr[dofs_y[node_ids]]
    out[:, 2] = q_arr[dofs_w[node_ids]]
    return out


def add_nodal_forces_to_rhs(rhs_vec, nodal_forces, node_ids, dofs_x, dofs_y, dofs_w):
    arr = rhs_vec.get_local()
    arr[dofs_x[node_ids]] += nodal_forces[:, 0]
    arr[dofs_y[node_ids]] += nodal_forces[:, 1]
    arr[dofs_w[node_ids]] += nodal_forces[:, 2]
    rhs_vec.set_local(arr)
    rhs_vec.apply("insert")


def local_project(v, Vout, u=None):
    dv = TrialFunction(Vout)
    v_ = TestFunction(Vout)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(Vout)
        solver.solve_local_rhs(u)
        return u
    solver.solve_local_rhs(u)
    return u


def build_output_displacement(q_fun, out_fun):
    # CHANGED:
    # Export a 3D displacement field assembled from plate variables so ParaView
    # and the fluid coupling still see a familiar vector displacement output.
    out_fun.assign(project(displacement_3d(q_fun), Vt))


sig = Function(Vsig, name="MembraneStress")
u_vis = Function(Vt, name="Displacement")

out_dir = "../results"
os.makedirs(out_dir, exist_ok=True)
xdmf_path = os.path.join(out_dir, "elastodynamics-results.xdmf")
xdmf_file = XDMFFile(xdmf_path)
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

print("Connecting solid to coupling server...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9000))
sock_file = sock.makefile("r")
sock.sendall((json.dumps({"role": "solid"}) + "\n").encode())
print("Solid connected.")

print("Building interface node sets...")
aero_node_ids, outer_surface_coords = get_aero_surface_node_ids()
cp_node_ids = extract_coupling_node_indices(
    n_span, n_chord, eta_chord_comm, outer_surface_coords
)
cp_nodes = outer_surface_coords[cp_node_ids, :]
np.savetxt(
    os.path.join(out_dir, "coupling_nodes.csv"),
    cp_nodes,
    delimiter=",",
    header="x,y,z",
    comments="",
)
print(f"Interface nodes ready: aero={len(aero_node_ids)}, coupling={len(cp_node_ids)}")

# CHANGED:
# Conservative transfer is still used, but now between fluid control points and
# plate midsurface nodes rather than solid boundary nodes.
interface_node_ids = aero_node_ids
interface_coords = outer_surface_coords[interface_node_ids, :]
print("Building local RBF transfer map...")
nbr_ids, nbr_w = build_local_rbf_map(cp_nodes, interface_coords, rbf_epsilon, n_neighbors=16)
print(
    f"Conservative mapping: {cp_nodes.shape[0]} fluid points <- "
    f"{interface_coords.shape[0]} solid interface nodes"
)

A_diag = np.ones((cp_nodes.shape[0],), dtype=float)
S_lumped = compute_S_lumped(len(interface_node_ids), nbr_ids, nbr_w, A_diag)
print(
    f"Operator setup: A=I ({len(A_diag)} dofs), "
    f"S_lumped min/max = {S_lumped.min():.3e}/{S_lumped.max():.3e}"
)

dofs_u_x = np.asarray(V.sub(0).sub(0).dofmap().dofs(), dtype=np.int64)
dofs_u_y = np.asarray(V.sub(0).sub(1).dofmap().dofs(), dtype=np.int64)
dofs_w = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)

if work_conservative_mode:
    t_aero.vector().zero()
    t_aero.vector().apply("insert")

u_cp0 = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
sock.sendall(
    (
        json.dumps(
            {
                "step": 0,
                "n_span": n_span,
                "n_chord": n_chord,
                "eta_span": eta_span_comm.tolist(),
                "eta_chord": eta_chord_comm.tolist(),
                "geometry": u_cp0,
            }
        )
        + "\n"
    ).encode()
)
print("Initial geometry sent.")

time = np.linspace(0.0, T, Nsteps + 1)
u_tip = np.zeros((Nsteps + 1,))
energies = np.zeros((Nsteps + 1, 4))
E_damp_acc = 0.0
force_relax = 1.0 if work_conservative_mode else 0.20
forces_prev = None
work_rel_errors = np.full((Nsteps,), np.nan, dtype=float)
work_Wf = np.full((Nsteps,), np.nan, dtype=float)
work_Ws = np.full((Nsteps,), np.nan, dtype=float)
ext_force_vec_template = q.vector().copy()
ext_force_vec_template.zero()

tip_x = x_leading_edge_at(span) + 0.75 * chord_at(span)
tip_y = span - 1.0e-8

for i in range(Nsteps):
    print(f"Solid step {i + 1}/{Nsteps}: waiting for force...")
    line = sock_file.readline()
    if line == "":
        raise RuntimeError("Coupling server disconnected while sending force data")

    data = json.loads(line)
    forces, used_2d_force = parse_force_payload(
        data, n_span, n_chord, eta_span_comm, eta_chord_comm
    )
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i + 1}")

    if i == 0:
        if used_2d_force:
            print(f"Solid: received 2D force payload ({n_span}x{n_chord})")
        else:
            print("Solid: received legacy spanwise force payload and remapped to 2D grid")

    if forces_prev is None:
        forces_eff = forces.copy()
    else:
        forces_eff = force_relax * forces + (1.0 - force_relax) * forces_prev
    forces_prev = forces_eff.copy()

    nodal_forces = None
    Fs_coeff = None
    if work_conservative_mode:
        Fs_coeff, nodal_forces = apply_Tf_operator(
            forces_eff, len(interface_node_ids), nbr_ids, nbr_w, A_diag, S_lumped
        )
        if not np.isfinite(nodal_forces).all():
            raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i + 1}")

        u_nodes_prev = get_nodal_displacements(
            q_old, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w
        )
        u_cp_prev = map_displacements_to_fluid(u_nodes_prev, nbr_ids, nbr_w)
        Wf = float(np.sum(u_cp_prev * (forces_eff * A_diag[:, None])))
        Ws = float(np.sum(u_nodes_prev * (Fs_coeff * S_lumped[:, None])))
        rel_work_err = abs(Wf - Ws) / max(abs(Wf), abs(Ws), 1.0e-16)
        work_rel_errors[i] = rel_work_err
        work_Wf[i] = Wf
        work_Ws[i] = Ws
        if i == 0 or (i + 1) % 20 == 0:
            print(
                f"Work audit step {i + 1}: "
                f"Wf={Wf:.6e}, Ws={Ws:.6e}, rel_err={rel_work_err:.3e}"
            )
    else:
        update_aero_traction(t_aero, forces_eff, n_span, n_chord, eta_chord_comm)

    # CHANGED:
    # Solve mixed plate step with Newton on the nonlinear residual.
    ext_force_vec = None
    if work_conservative_mode and nodal_forces is not None:
        ext_force_vec = ext_force_vec_template.copy()
        ext_force_vec.zero()
        add_nodal_forces_to_rhs(
            ext_force_vec, nodal_forces, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w
        )
    try:
        n_it, abs_res, rel_res = solve_nonlinear_step(
            q, jac_form, res, bcs, ext_force_vec=ext_force_vec
        )
    except RuntimeError as err:
        raise RuntimeError(f"Nonlinear plate solve failed at step {i + 1}/{Nsteps}: {err}")

    if i == 0 or (i + 1) % 20 == 0:
        print(
            f"Solid step {i + 1}/{Nsteps}: Newton converged in {n_it} iterations "
            f"(abs={abs_res:.3e}, rel={rel_res:.3e})"
        )

    update_fields(q, q_old, v_old, a_old)
    t = time[i + 1]

    build_output_displacement(q, u_vis)
    xdmf_file.write(u_vis, t)

    # CHANGED:
    # Stress output is now membrane stress on the plate midsurface rather than a
    # full 3D Cauchy stress tensor through the solid wing volume.
    q_mem, _q_w, _q_theta = q.split(deepcopy=True)
    local_project(membrane_stress(q_mem), Vsig, sig)
    xdmf_file.write(sig, t)

    E_elas = 0.5 * assemble(k_state(q_old, q_old))
    E_kin = 0.5 * assemble(m_state(v_old, v_old))
    E_damp_acc += dt_value * assemble(c_state(v_old, v_old))
    E_tot = E_elas + E_kin + E_damp_acc
    energies[i + 1, :] = np.array([E_elas, E_kin, E_damp_acc, E_tot])

    try:
        u_tip[i + 1] = q(tip_x, tip_y)[2]
    except RuntimeError:
        u_tip[i + 1] = q(tip_x, span - 1.0e-4)[2]

    if i < Nsteps - 1:
        u_nodes = get_nodal_displacements(q, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w)
        u_cp_arr = map_displacements_to_fluid(u_nodes, nbr_ids, nbr_w)
        msg_geo = json.dumps(
            {
                "step": i + 1,
                "n_span": n_span,
                "n_chord": n_chord,
                "eta_span": eta_span_comm.tolist(),
                "eta_chord": eta_chord_comm.tolist(),
                "geometry": u_cp_arr.tolist(),
            }
        )
        sock.sendall((msg_geo + "\n").encode())
        print(f"Solid step {i + 1}/{Nsteps}: geometry sent.")

sock_file.close()
sock.close()
print("Solid solver finished.")
print(f"Solid field outputs: {xdmf_path}")

# CHANGED:
# This diagnostic measures how close the Reissner-Mindlin solution is to the
# Kirchhoff-Love thin-plate constraint theta = grad(w).
_u_final, w_final, theta_final = q.split(deepcopy=True)
kl_gap_vec = kirchhoff_love_theta(w_final) - theta_final
kl_rotation_gap = project(
    sqrt(dot(kl_gap_vec, kl_gap_vec)),
    FunctionSpace(mesh, "CG", 1),
)
print(
    "Final Kirchhoff-Love compatibility diagnostic "
    f"(||grad(w)-theta||_L2) = {norm(kl_rotation_gap, 'l2'):.6e}"
)

if work_conservative_mode:
    valid = np.isfinite(work_rel_errors)
    if np.any(valid):
        errs = work_rel_errors[valid]
        final_err = float(errs[-1])
        max_err = float(np.max(errs))
        mean_err = float(np.mean(errs))
        w_tail = min(work_conv_window, len(errs))
        tail = errs[-w_tail:]
        tail_max = float(np.max(tail))
        tail_mean = float(np.mean(tail))
        conserved = tail_max <= work_rel_tol
        print(
            "Work conservation summary: "
            f"final={final_err:.3e}, mean={mean_err:.3e}, max={max_err:.3e}, "
            f"tail({w_tail}) mean={tail_mean:.3e}, tail({w_tail}) max={tail_max:.3e}, "
            f"tol={work_rel_tol:.1e}"
        )
        if conserved:
            print(f"Work conservation converged over last {w_tail} steps.")
        else:
            print(
                f"WARNING: Work conservation NOT converged over last {w_tail} steps "
                f"(tail max {tail_max:.3e} > tol {work_rel_tol:.1e})."
            )
    else:
        print("WARNING: No valid work-audit samples were collected.")

plt.figure()
plt.plot(time, u_tip)
plt.xlabel("Time")
plt.ylabel("Tip displacement")
tip_plot = os.path.join(out_dir, "tip_displacement.png")
plt.savefig(tip_plot, dpi=150)
plt.close()

plt.figure()
plt.plot(time, energies)
plt.legend(("elastic", "kinetic", "damping", "total"))
plt.xlabel("Time")
plt.ylabel("Energy")
energy_plot = os.path.join(out_dir, "energies.png")
plt.savefig(energy_plot, dpi=150)
plt.close()
print(f"Saved plots: {tip_plot}, {energy_plot}")
