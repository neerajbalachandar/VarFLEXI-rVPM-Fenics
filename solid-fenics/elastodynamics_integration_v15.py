# Chordwise and spanwise of v14
from dolfin import *
import json
import os
import socket

import numpy as np
from scipy.spatial import cKDTree

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


T = float(os.getenv("COUPLING_TTOT", "5.0"))
Nsteps = int(os.getenv("COUPLING_NSTEPS", "1000"))
dt_value = T / Nsteps
dt = Constant(dt_value)

span = float(os.getenv("SOLID_SPAN", "0.8"))
root_chord = float(os.getenv("SOLID_ROOT_CHORD", "0.12"))
tip_chord = float(os.getenv("SOLID_TIP_CHORD", "0.12"))
thickness_ratio = float(os.getenv("SOLID_THICKNESS_RATIO", "0.12"))
leading_edge_sweep = float(os.getenv("SOLID_LE_SWEEP", "0.0"))

nx = int(os.getenv("SOLID_NX", "12"))
ny = int(os.getenv("SOLID_NY", "240"))
nz = int(os.getenv("SOLID_NZ", "6"))

# Communication stations for the fluid panel grid.
n_span_comm = int(os.getenv("COUPLING_NSPAN_COMM", "80"))
n_chord_comm = int(os.getenv("COUPLING_NCHORD_COMM", "8"))
m_panels_comm = n_span_comm * n_chord_comm
span_sampling_mode = os.getenv("COUPLING_SPAN_SAMPLING", "node-stride").strip().lower()
span_custom_stride = os.getenv("COUPLING_SPAN_STRIDE")


def build_eta_span_comm(n_span_vals, ny_vals, mode):
    if n_span_vals < 1:
        raise ValueError("COUPLING_NSPAN_COMM must be >= 1")
    if n_span_vals == 1:
        return np.array([0.0], dtype=float), np.array([0], dtype=int)

    if mode == "midpoint":
        eta = (np.arange(n_span_vals, dtype=float) + 0.5) / n_span_vals
        idx = np.round(eta * ny_vals).astype(int)
        return eta, idx

    if mode == "node-stride":
        # Example: ny=200 and n_span=4 -> indices [0, 50, 100, 150].
        stride = max(1, ny_vals // n_span_vals)
        idx = np.arange(n_span_vals, dtype=int) * stride
        idx = np.clip(idx, 0, max(ny_vals, 1))
        if np.unique(idx).size < n_span_vals:
            idx = np.linspace(0, max(ny_vals - 1, 0), n_span_vals).round().astype(int)
        eta = idx.astype(float) / max(float(ny_vals), 1.0)
        return eta, idx

    if mode == "custom-stride":
        if span_custom_stride is None:
            raise ValueError(
                "COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE"
            )
        stride = int(span_custom_stride)
        if stride < 1:
            raise ValueError("COUPLING_SPAN_STRIDE must be >= 1")
        last_idx = (n_span_vals - 1) * stride
        if last_idx > ny_vals:
            raise ValueError(
                "Unsafe custom span stride: "
                f"(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = {last_idx} "
                f"exceeds SOLID_NY = {ny_vals}. Reduce stride/stations or increase SOLID_NY."
            )
        idx = np.arange(n_span_vals, dtype=int) * stride
        eta = idx.astype(float) / max(float(ny_vals), 1.0)
        return eta, idx

    if mode == "linspace":
        eta = np.linspace(0.0, 1.0, n_span_vals, endpoint=False)
        idx = np.round(eta * ny_vals).astype(int)
        return eta, idx

    raise ValueError(f"Unsupported COUPLING_SPAN_SAMPLING='{mode}'")


eta_span_comm, eta_span_comm_indices = build_eta_span_comm(n_span_comm, ny, span_sampling_mode)
eta_cp = float(os.getenv("COUPLING_ETA_CP", "0.75"))
if n_chord_comm <= 1:
    eta_chord_edges = np.array([0.0, 1.0], dtype=float)
    eta_chord_comm = np.array([eta_cp], dtype=float)
else:
    eta_chord_edges = np.linspace(0.0, 1.0, n_chord_comm + 1)
    eta_chord_comm = eta_chord_edges[:-1] + eta_cp * (
        eta_chord_edges[1:] - eta_chord_edges[:-1]
    )
eta_chord_le = eta_chord_edges[:-1].copy()
eta_chord_te = eta_chord_edges[1:].copy()

work_conservative_mode = True
rbf_radius = float(os.getenv("COUPLING_RBF_RADIUS", os.getenv("COUPLING_RBF_EPS", "0.08")))
rbf_neighbors = int(os.getenv("COUPLING_RBF_NEIGHBORS", "24"))
max_abs_force_component = float(os.getenv("COUPLING_MAX_FORCE_COMPONENT", "5.0e3"))

DEBUG_IO = os.getenv("COUPLING_DEBUG_IO", "0").strip().lower() not in ("0", "false", "no")
edge_eval_xi_eps = float(os.getenv("COUPLING_EDGE_EVAL_XI_EPS", "1.0e-6"))


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


def naca_half_thickness(xi):
    xi_clip = min(max(xi, 0.0), 1.0)
    return 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(xi_clip)
        - 0.1260 * xi_clip
        - 0.3516 * xi_clip ** 2
        + 0.2843 * xi_clip ** 3
        - 0.1015 * xi_clip ** 4
    )


mesh = BoxMesh(Point(0.0, 0.0, -1.5e-3), Point(1.0, span, 1.5e-3), nx, ny, nz)

coords = mesh.coordinates()
min_half_t = 0.10 * root_chord * thickness_ratio / max(nz, 1)
for i in range(coords.shape[0]):
    xi = min(max(coords[i, 0], 1.0e-4), 1.0)
    y_val = coords[i, 1]
    z_ref = coords[i, 2]
    chord = chord_at(y_val)
    x_le = x_leading_edge_at(y_val)
    zeta = 2.0 * z_ref
    half_t = max(chord * naca_half_thickness(xi), min_half_t)
    coords[i, 0] = x_le + xi * chord
    coords[i, 2] = zeta * half_t


def left(x, on_boundary):
    return near(x[1], 0.0) and on_boundary


facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)

panel_tol_x = 0.75 * (root_chord / max(nx, 1))
panel_tol_z = 1.25 * (root_chord * thickness_ratio / max(nz, 1))


class AeroSurface(SubDomain):
    def inside(self, X, on_boundary):
        if not on_boundary:
            return False
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            return False
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1.0e-12)
        if xi < -0.02 or xi > 1.02:
            return False
        z_surf = chord * naca_half_thickness(xi)
        return abs(abs(X[2]) - z_surf) <= panel_tol_z


aero_surface = AeroSurface()
aero_surface.mark(facet_markers, 5)
ds_aero = Measure("ds", domain=mesh, subdomain_data=facet_markers)

V = VectorFunctionSpace(mesh, "CG", 1)
Vt = VectorFunctionSpace(mesh, "CG", 1)
Vsig = TensorFunctionSpace(mesh, "DG", 0)

t_aero = Function(Vt, name="AerodynamicTraction")

E = float(os.getenv("SOLID_E", "6.8e10"))
nu = float(os.getenv("SOLID_NU", "0.35"))
mu = Constant(E / (2.0 * (1.0 + nu)))
lmbda = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
rho_s = float(os.getenv("SOLID_RHO", "1600.0"))
rho = Constant(rho_s)
eta_m = Constant(float(os.getenv("SOLID_ETA_M", "0.8")))
eta_k = Constant(float(os.getenv("SOLID_ETA_K", "1.0e-4")))
inext_penalty_chord_factor = float(
    os.getenv("SOLID_INEXT_PENALTY_CHORD_FACTOR", "25.0")
)
inext_penalty_span_factor = float(
    os.getenv("SOLID_INEXT_PENALTY_SPAN_FACTOR", "10.0")
)
enforce_chord_projection = os.getenv("COUPLING_ENFORCE_CHORD_PROJECTION", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
enforce_span_projection = os.getenv("COUPLING_ENFORCE_SPAN_PROJECTION", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
kappa_inext_chord = Constant(max(inext_penalty_chord_factor, 0.0) * E)
kappa_inext_span = Constant(max(inext_penalty_span_factor, 0.0) * E)
e_chord = Constant((1.0, 0.0, 0.0))
e_span = Constant((0.0, 1.0, 0.0))

alpha_m = Constant(0.10)
alpha_f = Constant(0.20)
gamma = Constant(0.5 + alpha_f - alpha_m)
beta = Constant((gamma + 0.5) ** 2 / 4.0)

print(
    f"Linear solid v15: span={span} m, c_root={root_chord} m, c_tip={tip_chord} m, "
    f"E={E:.3e} Pa, rho={rho_s} kg/m^3, comm_stations={n_span_comm}x{n_chord_comm}, "
    f"sampling={span_sampling_mode}"
)
print(f"Time setup: T={T} s, Nsteps={Nsteps}, dt={dt_value}")
print(
    f"Force transfer: compact Wendland C2 RBF, radius={rbf_radius:.4e} m, "
    f"neighbors={rbf_neighbors}"
)
print(
    f"Inextensibility controls: penalty(chord/span)=({inext_penalty_chord_factor:.2f},{inext_penalty_span_factor:.2f})*E, "
    f"coupling_chord_projection={enforce_chord_projection}, "
    f"coupling_span_projection={enforce_span_projection}"
)
if DEBUG_IO:
    print(f"Span comm node indices = {eta_span_comm_indices.tolist()}")
    print(f"Span comm etas         = {eta_span_comm.tolist()}")

du = TrialFunction(V)
u_ = TestFunction(V)
u = Function(V, name="Displacement")
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

zero = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, zero, left)


def sigma(r):
    eps = sym(grad(r))
    return 2.0 * mu * eps + lmbda * tr(eps) * Identity(len(r))


def m(u_trial, u_test):
    return rho * inner(u_trial, u_test) * dx


def k(u_trial, u_test):
    base = inner(sigma(u_trial), sym(grad(u_test))) * dx

    # Penalize extensional strain along chord/span directions to reduce
    # artificial panel stretching in the coupled geometry transfer.
    eps_trial = sym(grad(u_trial))
    eps_test = sym(grad(u_test))
    gct = dot(e_chord, eps_trial * e_chord)
    gcs = dot(e_chord, eps_test * e_chord)
    gst = dot(e_span, eps_trial * e_span)
    gss = dot(e_span, eps_test * e_span)
    pen = kappa_inext_chord * gct * gcs * dx + kappa_inext_span * gst * gss * dx
    return base + pen


def c(u_trial, u_test):
    return eta_m * m(u_trial, u_test) + eta_k * k(u_trial, u_test)


def Wext(u_test):
    return dot(u_test, t_aero) * ds_aero(5)


def update_a(u_new, u_prev, v_prev, a_prev, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u_new - u_prev - dt_ * v_prev) / beta_ / dt_ ** 2 - (
        1.0 - 2.0 * beta_
    ) / (2.0 * beta_) * a_prev


def update_v(a_new, u_prev, v_prev, a_prev, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_prev + dt_ * ((1.0 - gamma_) * a_prev + gamma_ * a_new)


def update_fields(u_fun, u_prev, v_prev, a_prev):
    u_vec = u_fun.vector()
    u0_vec = u_prev.vector()
    v0_vec = v_prev.vector()
    a0_vec = a_prev.vector()

    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    v_prev.vector()[:] = v_vec
    a_prev.vector()[:] = a_vec
    u_prev.vector()[:] = u_vec


def avg(x_old, x_new, alpha):
    return alpha * x_old + (1.0 - alpha) * x_new


a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

res = (
    m(avg(a_old, a_new, alpha_m), u_)
    + c(avg(v_old, v_new, alpha_f), u_)
    + k(avg(u_old, du, alpha_f), u_)
    - Wext(u_)
)

a_form = lhs(res)
L_form = rhs(res)
K, _ = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True


def local_project(v_expr, Vout, u_out=None):
    dv = TrialFunction(Vout)
    v_test = TestFunction(Vout)
    a_proj = inner(dv, v_test) * dx
    b_proj = inner(v_expr, v_test) * dx
    solver_local = LocalSolver(a_proj, b_proj)
    solver_local.factorize()
    if u_out is None:
        u_out = Function(Vout)
        solver_local.solve_local_rhs(u_out)
        return u_out
    solver_local.solve_local_rhs(u_out)
    return u_out


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
    for c_idx in range(vals_src.shape[1]):
        out[:, c_idx] = np.interp(x_dst, x_src, vals_src[:, c_idx])
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
    for j_idx in range(n_chord_in):
        grid_span[:, j_idx, :] = interp_profile(
            eta_span_in, grid_in[:, j_idx, :], eta_span_out
        )

    grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
    for i_idx in range(n_span_out):
        grid_out[i_idx, :, :] = interp_profile(
            eta_chord_in, grid_span[i_idx, :, :], eta_chord_out
        )

    return grid_out.reshape((n_out, 3))


def parse_force_payload(data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    if "n_span" in data and "n_chord" in data and "force" in data:
        n_span_in = int(data.get("n_span", 0))
        n_chord_in = int(data.get("n_chord", 0))
        force_raw = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        indexing = str(data.get("indexing", "span-major"))
        if n_span_in > 0 and n_chord_in > 0 and len(force_raw) == n_span_in * n_chord_in:
            force_grid = np.zeros((n_span_in, n_chord_in, 3), dtype=float)
            k_idx = 0
            if indexing == "span-major":
                for i_idx in range(n_span_in):
                    for j_idx in range(n_chord_in):
                        force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                        k_idx += 1
            elif indexing == "chord-major":
                for j_idx in range(n_chord_in):
                    for i_idx in range(n_span_in):
                        force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                        k_idx += 1
            else:
                raise RuntimeError(f"Unsupported force indexing '{indexing}'")
            force_raw = force_grid.reshape((-1, 3))
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
    v_scalar = Vt.sub(0).collapse()
    coords_v = v_scalar.tabulate_dof_coordinates().reshape((-1, 3))
    ids = []
    for i_node, X in enumerate(coords_v):
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            continue
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1.0e-12)
        if xi < -0.02 or xi > 1.02:
            continue
        z_surf = chord * naca_half_thickness(xi)
        if abs(abs(X[2]) - z_surf) > panel_tol_z:
            continue
        ids.append(i_node)
    return np.asarray(sorted(set(ids)), dtype=np.int64), coords_v


def build_span_chord_targets(eta_span_vals, eta_chord_vals, xi_eps=0.0):
    pts = np.zeros((len(eta_span_vals) * len(eta_chord_vals), 3), dtype=float)
    k_idx = 0
    for eta_s in eta_span_vals:
        y_val = eta_s * span
        chord = chord_at(y_val)
        x_le = x_leading_edge_at(y_val)
        for eta_c in eta_chord_vals:
            xi_eff = float(np.clip(eta_c, 0.0, 1.0))
            if xi_eff <= 0.0:
                xi_eff = min(1.0, xi_eff + xi_eps)
            elif xi_eff >= 1.0:
                xi_eff = max(0.0, xi_eff - xi_eps)
            pts[k_idx, 0] = x_le + xi_eff * chord
            pts[k_idx, 1] = y_val
            pts[k_idx, 2] = 0.0
            k_idx += 1
    return pts


def sample_vector_field_at_targets(u_fun, targets_xyz, fallback_tree=None, fallback_vals=None):
    out = np.zeros((targets_xyz.shape[0], 3), dtype=float)
    for k_idx in range(targets_xyz.shape[0]):
        pt = Point(
            float(targets_xyz[k_idx, 0]),
            float(targets_xyz[k_idx, 1]),
            float(targets_xyz[k_idx, 2]),
        )
        try:
            val = u_fun(pt)
            out[k_idx, 0] = float(val[0])
            out[k_idx, 1] = float(val[1])
            out[k_idx, 2] = float(val[2])
        except RuntimeError:
            if fallback_tree is None or fallback_vals is None:
                continue
            _, idx = fallback_tree.query(targets_xyz[k_idx, :], k=1)
            out[k_idx, :] = fallback_vals[int(idx), :]
    return out


def project_le_te_inextensible(u_le_arr, u_te_arr, le_ref, te_ref):
    # Enforce reference chord length at each span station in communicated geometry.
    Xle = le_ref + u_le_arr
    Xte = te_ref + u_te_arr
    mid = 0.5 * (Xle + Xte)

    ref_vec = te_ref - le_ref
    ref_len = np.linalg.norm(ref_vec, axis=1)

    cur_vec = Xte - Xle
    cur_len = np.linalg.norm(cur_vec, axis=1)
    eps = 1.0e-14

    ref_dir = np.zeros_like(ref_vec)
    ok_ref = ref_len > eps
    ref_dir[ok_ref, :] = ref_vec[ok_ref, :] / ref_len[ok_ref, None]
    if np.any(~ok_ref):
        ref_dir[~ok_ref, :] = np.array([1.0, 0.0, 0.0], dtype=float)

    cur_dir = np.zeros_like(cur_vec)
    ok_cur = cur_len > eps
    cur_dir[ok_cur, :] = cur_vec[ok_cur, :] / cur_len[ok_cur, None]
    if np.any(~ok_cur):
        cur_dir[~ok_cur, :] = ref_dir[~ok_cur, :]

    Xle_p = mid - 0.5 * ref_len[:, None] * cur_dir
    Xte_p = mid + 0.5 * ref_len[:, None] * cur_dir
    u_le_p = Xle_p - le_ref
    u_te_p = Xte_p - te_ref
    return u_le_p, u_te_p, cur_len, ref_len


def project_spanwise_inextensible_line(u_line, ref_line):
    # Enforce reference segment lengths along a spanwise polyline (LE or TE).
    X = ref_line + u_line
    n_pts = X.shape[0]
    if n_pts <= 1:
        return u_line, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    seg_ref_vec = ref_line[1:, :] - ref_line[:-1, :]
    seg_ref_len = np.linalg.norm(seg_ref_vec, axis=1)
    seg_cur_vec = X[1:, :] - X[:-1, :]
    seg_cur_len = np.linalg.norm(seg_cur_vec, axis=1)

    Xp = np.zeros_like(X)
    Xp[0, :] = X[0, :]
    eps = 1.0e-14
    for i in range(1, n_pts):
        v = X[i, :] - X[i - 1, :]
        lv = np.linalg.norm(v)
        if lv > eps:
            d = v / lv
        else:
            rv = ref_line[i, :] - ref_line[i - 1, :]
            lrv = np.linalg.norm(rv)
            if lrv > eps:
                d = rv / lrv
            else:
                d = np.array([0.0, 1.0, 0.0], dtype=float)
        Xp[i, :] = Xp[i - 1, :] + seg_ref_len[i - 1] * d

    u_proj = Xp - ref_line
    return u_proj, seg_cur_len, seg_ref_len


def build_local_rbf_map(fluid_points, solid_points, radius, n_neighbors=24):
    fluid_points = np.asarray(fluid_points, dtype=float)
    solid_points = np.asarray(solid_points, dtype=float)
    support_radius = float(radius)
    if support_radius <= 0.0:
        raise ValueError("COUPLING_RBF_RADIUS must be positive for compact RBF transfer")
    n_s = solid_points.shape[0]
    k = int(max(1, min(n_neighbors, n_s)))

    tree = cKDTree(solid_points)
    dists, idx = tree.query(fluid_points, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    nbr_ids = idx.astype(np.int64)

    # Compact-support Wendland C2 kernel:
    #   phi(r) = (1 - q)^4 (4q + 1), q = r / R, for q < 1
    #   phi(r) = 0,                         for q >= 1
    # This keeps force transfer local and shuts off exactly outside radius R.
    q = dists / support_radius
    nbr_w = np.zeros_like(dists, dtype=float)
    inside = q < 1.0
    one_minus_q = 1.0 - q[inside]
    nbr_w[inside] = (one_minus_q ** 4) * (4.0 * q[inside] + 1.0)

    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
    for bi in bad:
        # If the radius is too small for this CP, keep the map usable by assigning
        # the full CP force to the nearest solid node.
        nbr_w[bi, :] = 0.0
        nbr_w[bi, 0] = 1.0
    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    nbr_w /= np.maximum(row_sum, 1.0e-16)
    return nbr_ids, nbr_w


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


def add_nodal_forces_to_rhs(rhs_vec, nodal_forces, node_ids, dofs_x, dofs_y, dofs_z):
    arr = rhs_vec.get_local()
    arr[dofs_x[node_ids]] += nodal_forces[:, 0]
    arr[dofs_y[node_ids]] += nodal_forces[:, 1]
    arr[dofs_z[node_ids]] += nodal_forces[:, 2]
    rhs_vec.set_local(arr)
    rhs_vec.apply("insert")


def get_nodal_displacements(u_fun, node_ids, dofs_x, dofs_y, dofs_z):
    u_arr = u_fun.vector().get_local()
    out = np.zeros((len(node_ids), 3), dtype=float)
    out[:, 0] = u_arr[dofs_x[node_ids]]
    out[:, 1] = u_arr[dofs_y[node_ids]]
    out[:, 2] = u_arr[dofs_z[node_ids]]
    return out


sig = Function(Vsig, name="CauchyStress")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_dir = os.path.join(repo_root, "results", "solid", "v15")
os.makedirs(out_dir, exist_ok=True)
xdmf_path = os.path.join(out_dir, "elastodynamics-results.xdmf")
xdmf_file = XDMFFile(xdmf_path)
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

mesh_pvd_path = os.path.join(out_dir, "solid_mesh.pvd")
u_pvd_path = os.path.join(out_dir, "solid_displacement.pvd")
sig_pvd_path = os.path.join(out_dir, "solid_stress.pvd")
mesh_pvd = File(mesh_pvd_path)
u_pvd = File(u_pvd_path)
sig_pvd = File(sig_pvd_path)
mesh_pvd << mesh

print("Connecting solid to coupling server...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(
    (
        os.getenv("COUPLING_HOST", "127.0.0.1"),
        int(os.getenv("COUPLING_PORT", "9000")),
    )
)
sock_file = sock.makefile("r")
sock.sendall((json.dumps({"role": "solid"}) + "\n").encode())
print("Solid connected.")

print("Building interface sets and communication targets...")
aero_node_ids, aero_coords = get_aero_surface_node_ids()
interface_node_ids = aero_node_ids
interface_coords = aero_coords[interface_node_ids, :]
interface_tree = cKDTree(interface_coords)

cp_targets = build_span_chord_targets(eta_span_comm, eta_chord_comm, xi_eps=0.0)
le_targets = build_span_chord_targets(eta_span_comm, eta_chord_le, xi_eps=edge_eval_xi_eps)
te_targets = build_span_chord_targets(eta_span_comm, eta_chord_te, xi_eps=edge_eval_xi_eps)

np.savetxt(
    os.path.join(out_dir, "coupling_cp_targets.csv"),
    cp_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)
np.savetxt(
    os.path.join(out_dir, "coupling_le_targets.csv"),
    le_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)
np.savetxt(
    os.path.join(out_dir, "coupling_te_targets.csv"),
    te_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)
with open(os.path.join(out_dir, "coupling_targets_indexing.csv"), "w") as fp:
    fp.write("k,i_span,j_chord,x_cp,y_cp,z_cp,x_le,y_le,z_le,x_te,y_te,z_te\n")
    k_idx = 0
    for i_idx in range(n_span_comm):
        for j_idx in range(n_chord_comm):
            fp.write(
                f"{k_idx},{i_idx},{j_idx},"
                f"{cp_targets[k_idx,0]:.12e},{cp_targets[k_idx,1]:.12e},{cp_targets[k_idx,2]:.12e},"
                f"{le_targets[k_idx,0]:.12e},{le_targets[k_idx,1]:.12e},{le_targets[k_idx,2]:.12e},"
                f"{te_targets[k_idx,0]:.12e},{te_targets[k_idx,1]:.12e},{te_targets[k_idx,2]:.12e}\n"
            )
            k_idx += 1

nbr_ids, nbr_w = build_local_rbf_map(
    cp_targets, interface_coords, rbf_radius, n_neighbors=rbf_neighbors
)
A_diag = np.ones((cp_targets.shape[0],), dtype=float)
S_lumped = compute_S_lumped(len(interface_node_ids), nbr_ids, nbr_w, A_diag)

print(
    f"Interface ready: surface_nodes={len(interface_node_ids)}, "
    f"cp_stations={len(cp_targets)}, RBF neighbors={nbr_ids.shape[1]}"
)

dofs_x = np.asarray(V.sub(0).dofmap().dofs(), dtype=np.int64)
dofs_y = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)
dofs_z = np.asarray(V.sub(2).dofmap().dofs(), dtype=np.int64)

ext_force_vec_template = u.vector().copy()
ext_force_vec_template.zero()

time = np.linspace(0.0, T, Nsteps + 1)
u_tip = np.zeros((Nsteps + 1,), dtype=float)
energies = np.zeros((Nsteps + 1, 4), dtype=float)
E_damp_acc = 0.0
force_relax = 1.0
forces_prev = None
work_rel_errors = np.full((Nsteps,), np.nan, dtype=float)
work_Wf = np.full((Nsteps,), np.nan, dtype=float)
work_Ws = np.full((Nsteps,), np.nan, dtype=float)

tip_x = x_leading_edge_at(span) + eta_cp * chord_at(span)
tip_y = span - 1.0e-8
tip_z = 0.0

zero_payload = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
init_msg = {
    "step": 0,
    "dt": dt_value,
    "ttot": T,
    "nsteps": Nsteps,
    "n_span": n_span_comm,
    "n_chord": n_chord_comm,
    "indexing": "span-major",
    "eta_span": eta_span_comm.tolist(),
    "eta_chord": eta_chord_comm.tolist(),
    "geometry": zero_payload,
    "geometry_le": zero_payload,
    "geometry_te": zero_payload,
    "rotation": zero_payload,
    "rotation_le": zero_payload,
    "rotation_te": zero_payload,
}
sock.sendall((json.dumps(init_msg) + "\n").encode())
print("Initial zero geometry sent.")

xdmf_file.write(u, 0.0)
local_project(sigma(u), Vsig, sig)
xdmf_file.write(sig, 0.0)
u_pvd << (u, 0.0)
sig_pvd << (sig, 0.0)

for i_step in range(Nsteps):
    print(f"Solid step {i_step + 1}/{Nsteps}: waiting for force...")
    line = sock_file.readline()
    if line == "":
        raise RuntimeError("Coupling server disconnected while sending force data")

    data = json.loads(line)
    forces, used_structured_force = parse_force_payload(
        data, n_span_comm, n_chord_comm, eta_span_comm, eta_chord_comm
    )
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i_step + 1}")

    if i_step == 0:
        if used_structured_force:
            print(
                f"Solid: received structured force payload ({n_span_comm}x{n_chord_comm})"
            )
        else:
            print("Solid: received legacy force payload and resampled to coupling stations")

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
            raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i_step + 1}")

    b = assemble(L_form)
    if nodal_forces is not None:
        ext_force_vec = ext_force_vec_template.copy()
        ext_force_vec.zero()
        add_nodal_forces_to_rhs(
            ext_force_vec, nodal_forces, interface_node_ids, dofs_x, dofs_y, dofs_z
        )
        b.axpy(1.0, ext_force_vec)
    bc.apply(b)
    solver.solve(u.vector(), b)

    if work_conservative_mode and nodal_forces is not None and Fs_coeff is not None:
        interface_disp_prev = get_nodal_displacements(
            u_old, interface_node_ids, dofs_x, dofs_y, dofs_z
        )
        u_cp_prev = sample_vector_field_at_targets(
            u_old,
            cp_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_prev,
        )
        Wf = float(np.sum(u_cp_prev * (forces_eff * A_diag[:, None])))
        Ws = float(np.sum(interface_disp_prev * (Fs_coeff * S_lumped[:, None])))
        rel_work_err = abs(Wf - Ws) / max(abs(Wf), abs(Ws), 1.0e-16)
        work_rel_errors[i_step] = rel_work_err
        work_Wf[i_step] = Wf
        work_Ws[i_step] = Ws
        if i_step == 0 or (i_step + 1) % 20 == 0:
            print(
                f"Work audit step {i_step + 1}: "
                f"Wf={Wf:.6e}, Ws={Ws:.6e}, rel_err={rel_work_err:.3e}"
            )

    update_fields(u, u_old, v_old, a_old)
    t = time[i_step + 1]

    xdmf_file.write(u, t)
    u_pvd << (u, float(t))
    local_project(sigma(u), Vsig, sig)
    xdmf_file.write(sig, t)
    sig_pvd << (sig, float(t))

    E_elas = 0.5 * assemble(k(u_old, u_old))
    E_kin = 0.5 * assemble(m(v_old, v_old))
    E_damp_acc += dt_value * assemble(c(v_old, v_old))
    E_tot = E_elas + E_kin + E_damp_acc
    energies[i_step + 1, :] = np.array([E_elas, E_kin, E_damp_acc, E_tot])

    try:
        u_tip[i_step + 1] = u(Point(tip_x, tip_y, tip_z))[2]
    except RuntimeError:
        u_tip[i_step + 1] = 0.0

    if i_step < Nsteps - 1:
        interface_disp_cur = get_nodal_displacements(
            u, interface_node_ids, dofs_x, dofs_y, dofs_z
        )#gets the diaplcements at the stations which it has to send to the fluid solver
        u_le_arr = sample_vector_field_at_targets(
            u,
            le_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_cur,
        )
        u_te_arr = sample_vector_field_at_targets(
            u,
            te_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_cur,
        )
        if enforce_chord_projection:
            u_le_arr, u_te_arr, chord_len_cur, chord_len_ref = project_le_te_inextensible(
                u_le_arr, u_te_arr, le_targets, te_targets
            )
            if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                rel_ch = np.abs(chord_len_cur - chord_len_ref) / np.maximum(chord_len_ref, 1.0e-14)
                print(
                    f"Chord projection step {i_step+1}: "
                    f"pre-proj rel chord err max/mean = {np.max(rel_ch):.3e}/{np.mean(rel_ch):.3e}"
                )
        if enforce_span_projection:
            u_le_grid = u_le_arr.reshape((n_span_comm, n_chord_comm, 3))
            u_te_grid = u_te_arr.reshape((n_span_comm, n_chord_comm, 3))
            le_ref_grid = le_targets.reshape((n_span_comm, n_chord_comm, 3))
            te_ref_grid = te_targets.reshape((n_span_comm, n_chord_comm, 3))
            rel_sp_le_all = []
            rel_sp_te_all = []
            for j_idx in range(n_chord_comm):
                u_le_col, span_len_cur_le, span_len_ref_le = project_spanwise_inextensible_line(
                    u_le_grid[:, j_idx, :], le_ref_grid[:, j_idx, :]
                )
                u_te_col, span_len_cur_te, span_len_ref_te = project_spanwise_inextensible_line(
                    u_te_grid[:, j_idx, :], te_ref_grid[:, j_idx, :]
                )
                u_le_grid[:, j_idx, :] = u_le_col
                u_te_grid[:, j_idx, :] = u_te_col
                if span_len_ref_le.size > 0:
                    rel_sp_le = np.abs(span_len_cur_le - span_len_ref_le) / np.maximum(
                        span_len_ref_le, 1.0e-14
                    )
                    rel_sp_te = np.abs(span_len_cur_te - span_len_ref_te) / np.maximum(
                        span_len_ref_te, 1.0e-14
                    )
                    rel_sp_le_all.append(rel_sp_le)
                    rel_sp_te_all.append(rel_sp_te)
            u_le_arr = u_le_grid.reshape((-1, 3))
            u_te_arr = u_te_grid.reshape((-1, 3))
            if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                if len(rel_sp_le_all) > 0:
                    rel_sp_le = np.concatenate(rel_sp_le_all)
                    rel_sp_te = np.concatenate(rel_sp_te_all)
                    print(
                        f"Span projection step {i_step+1}: "
                        f"LE pre-proj rel seg err max/mean = {np.max(rel_sp_le):.3e}/{np.mean(rel_sp_le):.3e}, "
                        f"TE pre-proj rel seg err max/mean = {np.max(rel_sp_te):.3e}/{np.mean(rel_sp_te):.3e}"
                    )
        # Keep CP payload consistent with communicated LE/TE edge displacements.
        u_le_grid = u_le_arr.reshape((n_span_comm, n_chord_comm, 3))
        u_te_grid = u_te_arr.reshape((n_span_comm, n_chord_comm, 3))
        d_eta = np.maximum(eta_chord_te - eta_chord_le, 1.0e-14)
        cp_w = (eta_chord_comm - eta_chord_le) / d_eta
        cp_w = np.clip(cp_w, 0.0, 1.0)
        u_cp_grid = np.zeros_like(u_le_grid)
        for j_idx in range(n_chord_comm):
            w = cp_w[j_idx]
            u_cp_grid[:, j_idx, :] = (1.0 - w) * u_le_grid[:, j_idx, :] + w * u_te_grid[:, j_idx, :]
        u_cp_arr = u_cp_grid.reshape((-1, 3))
        zero_rot = np.zeros_like(u_cp_arr)

        if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
            print(f"SEND step {i_step + 1} first LE/TE = {u_le_arr[0, :].tolist()} / {u_te_arr[0, :].tolist()}")
            print(f"SEND step {i_step + 1} last  LE/TE = {u_le_arr[-1, :].tolist()} / {u_te_arr[-1, :].tolist()}")

        msg_geo = json.dumps(
            {
                "step": i_step + 1,
                "dt": dt_value,
                "ttot": T,
                "nsteps": Nsteps,
                "n_span": n_span_comm,
                "n_chord": n_chord_comm,
                "indexing": "span-major",
                "eta_span": eta_span_comm.tolist(),
                "eta_chord": eta_chord_comm.tolist(),
                "geometry": u_cp_arr.tolist(),
                "geometry_le": u_le_arr.tolist(),
                "geometry_te": u_te_arr.tolist(),
                "rotation": zero_rot.tolist(),
                "rotation_le": zero_rot.tolist(),
                "rotation_te": zero_rot.tolist(),
            }
        )
        sock.sendall((msg_geo + "\n").encode())
        print(f"Solid step {i_step + 1}/{Nsteps}: geometry sent.")

sock_file.close()
sock.close()
print("Solid solver finished.")
print(f"Solid field outputs: {xdmf_path}")
print(f"Solid VTK outputs: {u_pvd_path}, {sig_pvd_path}, {mesh_pvd_path}")

diag_csv = os.path.join(out_dir, "solid_v15_diagnostics.csv")
with open(diag_csv, "w") as fp:
    fp.write("step,time,u_tip,E_elas,E_kin,E_damp,E_tot,work_Wf,work_Ws,work_rel_error\n")
    for k_idx in range(Nsteps):
        fp.write(
            f"{k_idx + 1},{time[k_idx + 1]:.12e},{u_tip[k_idx + 1]:.12e},"
            f"{energies[k_idx + 1, 0]:.12e},{energies[k_idx + 1, 1]:.12e},"
            f"{energies[k_idx + 1, 2]:.12e},{energies[k_idx + 1, 3]:.12e},"
            f"{work_Wf[k_idx]:.12e},{work_Ws[k_idx]:.12e},{work_rel_errors[k_idx]:.12e}\n"
        )

print(f"Diagnostics: {diag_csv}")
