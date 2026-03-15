# Airfoil geometry with chordwise/spanwise coupling, work-conservative transfer,
# and SI-unit parameterization.
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import socket
import json
import os
from scipy.spatial import cKDTree

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

T = 4.0                     # s
Nsteps = 200
dt_value = T/Nsteps
dt = Constant(dt_value)

span = 1.5                  # m
root_chord = 0.20           # m
tip_chord = 0.12            # m
thickness_ratio = 0.12
leading_edge_sweep = 0.0

# Speed-tuned mesh for interactive coupled runs.
nx, ny, nz = 12, 120, 6

mesh = BoxMesh(Point(0.0, 0.0, -0.5), Point(1.0, span, 0.5), nx, ny, nz)
# span_strips = 200

# Match fluid panels - rectangular meshes, projected meshes
n_span = 80
n_chord = 12


# Solid sends geometry with n_span, n_chord, eta_span, eta_chordas flattened 1D arrays, two for locating and two for parametrizing
m_panels_comm = n_span * n_chord
eta_span_comm = np.linspace(0.0, 1.0, n_span)

# expand...
eta_chord_edges = np.linspace(0.0, 1.0, n_chord + 1)
eta_chord_comm = eta_chord_edges[:-1] + 0.75 * (eta_chord_edges[1:] - eta_chord_edges[:-1])

# Conservative coupling controls
work_conservative_mode = True # Can we use both at once, relaxation and conservation?
rbf_epsilon = 0.06 # shape parameter of local Gaussian RBF for conservative mapping
aoa_deg = 8.0 # deg

# Prescribed root flapping motion (about x-axis at y=0)
flap_amp_deg = 40.0
flap_freq_hz = 2.0
flap_phase = 0.0
flap_ramp_time = 0.5  # s, smooth ramp-in to avoid a hard first-step jump


def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


# So if we are extending to anti-symmetric airfoils, how can we do it?
def naca_half_thickness(xi):
    xi_clip = min(max(xi, 0.0), 1.0)
    return 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(xi_clip)
        - 0.1260 * xi_clip
        - 0.3516 * xi_clip**2
        + 0.2843 * xi_clip**3
        - 0.1015 * xi_clip**4
    )


# Projecting the cuboidal mesh to the airfoil
coords = mesh.coordinates()
min_half_t = 0.10 * root_chord * thickness_ratio / nz
for i in range(coords.shape[0]):
    # Keep xi away from exactly 0 to avoid zero-thickness leading-edge collapse.
    xi = min(max(coords[i, 0], 1.0e-4), 1.0)
    y_val = coords[i, 1]
    z_ref = coords[i, 2]
    chord = chord_at(y_val)
    x_le = x_leading_edge_at(y_val)
    zeta = 2.0 * z_ref
    half_t = max(chord * naca_half_thickness(xi), min_half_t)
    coords[i, 0] = x_le + xi * chord
    coords[i, 2] = zeta * half_t

# Left and right boundaries defined. Here, in this function x = [x,y,z]
def left(x, on_boundary):
    return near(x[1], 0.) and on_boundary

def right(x, on_boundary):
    return near(x[1], span) and on_boundary

# integer tags for 2D facets of 3D mesh, are used to define ds_aero and integrate only on that boundary
# Marker-based traction path - inbuilt Fenics
facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)

# y is handled using binning/projection, so every node gets assigned to a span bin directly
# Geometric tolerances used to decide whether a mesh node is “close enough” to your aerodynamic surface/panel target locations, for x and z
panel_tol_x = 0.75 * (root_chord / nx)
panel_tol_z = 1.25 * (root_chord * thickness_ratio / nz)

# AeroSurface is a subdomain wherever you want to give the aeroforce
class AeroSurface(SubDomain):
    def inside(self, X, on_boundary):
        if not on_boundary:
            return False
        y_val = X[1] 
        chord = chord_at(y_val) # chord length at given y_val
        if chord <= 0.0:
            return False
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / chord
        if xi < -0.02 or xi > 1.02: # 0.02 tolerance, why is panel_tol_x not used here?
            return False
        z_surf = chord * naca_half_thickness(xi)
        return abs(abs(X[2]) - z_surf) <= panel_tol_z

aero_surface = AeroSurface()
aero_surface.mark(facet_markers, 5) # All the facets (2D grid) surrounding the boundary are marked 5 - basically the surfaces of the airfoil

ds_aero = Measure("ds", domain=mesh, subdomain_data=facet_markers) # In Wext, only that boundary gets aerodynamic traction contribution.

V = VectorFunctionSpace(mesh, "CG", 1)
Vt = VectorFunctionSpace(mesh, "CG", 1)
Vsig = TensorFunctionSpace(mesh, "DG", 0)

t_aero = Function(Vt, name="AerodynamicTraction")

# Problem Defn (SI)
E = 7.0e9
nu = 0.33
mu = Constant(E/(2.0*(1.0+nu)))
lmbda = Constant(E*nu/((1.0+nu)*(1.0-2.0*nu)))
rho_s = 1600.0               # kg/m^3
rho = Constant(rho_s)
eta_m = Constant(0.8)        # 1/s (Rayleigh mass coefficient)
eta_k = Constant(1.0e-4)     # s   (Rayleigh stiffness coefficient)

alpha_m = Constant(0.10)
alpha_f = Constant(0.20)
gamma = Constant(0.5+alpha_f-alpha_m)
beta = Constant((gamma+0.5)**2/4.)

print(
    f"SI setup (solid): span={span} m, c_root={root_chord} m, c_tip={tip_chord} m, "
    f"E={E:.3e} Pa, rho_s={rho_s} kg/m^3, AoA={aoa_deg} deg"
)

du_trial = TrialFunction(V)
u_ = TestFunction(V)
u = Function(V, name="Displacement")
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

class RootFlapBC(UserExpression):
    def __init__(self, theta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.theta = float(theta)

    def eval(self, values, x):
        cth = np.cos(self.theta)
        sth = np.sin(self.theta)
        y0 = x[1]
        z0 = x[2]
        values[0] = 0.0
        values[1] = y0 * cth - z0 * sth - y0
        values[2] = y0 * sth + z0 * cth - z0

    def value_shape(self):
        return (3,)

flap_amp = flap_amp_deg * np.pi / 180.0
flap_bc_expr = RootFlapBC(theta=0.0, degree=1)
bc = DirichletBC(V, flap_bc_expr, left)
bc_hom = DirichletBC(V, Constant((0.0, 0.0, 0.0)), left)

print(
    f"Flapping BC: amp={flap_amp_deg} deg, freq={flap_freq_hz} Hz, phase={flap_phase} rad"
)

def sigma_lin(r):
    return 2.0 * mu * sym(grad(r)) + lmbda * tr(sym(grad(r))) * Identity(len(r))

def m_form(u_val, u_test):
    return rho * inner(u_val, u_test) * dx

def c_form(u_val, u_test):
    return eta_m * m_form(u_val, u_test) + eta_k * inner(sigma_lin(u_val), sym(grad(u_test))) * dx

def Wext(u_test):
    return dot(u_test, t_aero) * ds_aero(5)

def psi_hyperelastic(w):
    I = Identity(mesh.geometry().dim())
    F = I + grad(w)
    C = F.T * F
    Ic = tr(C)
    J = det(F)
    J_safe = conditional(gt(J, 1.0e-9), J, 1.0e-9)
    return (mu / 2.0) * (Ic - 3.0 - 2.0 * ln(J_safe)) + (lmbda / 2.0) * (ln(J_safe) ** 2)

def sigma_cauchy(w):
    I = Identity(mesh.geometry().dim())
    F = I + grad(w)
    b = F * F.T
    J = det(F)
    J_safe = conditional(gt(J, 1.0e-9), J, 1.0e-9)
    return (mu / J_safe) * (b - I) + (lmbda * ln(J_safe) / J_safe) * I

def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

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
    return alpha*x_old + (1-alpha)*x_new

a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

u_alpha = avg(u_old, u, alpha_f)

Pi_int = psi_hyperelastic(u_alpha) * dx
res = m_form(avg(a_old, a_new, alpha_m), u_) \
    + c_form(avg(v_old, v_new, alpha_f), u_) \
    + derivative(Pi_int, u, u_) \
    - Wext(u_)
jac_form = derivative(res, u, du_trial)

# What is this? and should we try printing it?
panel_node_cache = {}
coupling_node_cache = {}
max_abs_force_component = 5.0e3


# Ensuring parametric eta works fine \in [0,1], and normalization - Returns eta as 1D array
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
# Isn't it already normalised? What advantage is giving?


# 
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


def resample_forces_to_shape(forces, n_span_out, n_chord_out,
                             eta_span_in=None, eta_chord_in=None,
                             eta_span_out=None, eta_chord_out=None):
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
        grid_span[:, j, :] = interp_profile(eta_span_in, grid_in[:, j, :], eta_span_out)

    grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
    for i in range(n_span_out):
        grid_out[i, :, :] = interp_profile(eta_chord_in, grid_span[i, :, :], eta_chord_out)

    return grid_out.reshape((n_out, 3))


def get_panel_node_ids(n_span, n_chord, eta_chord):
    eta_chord = as_eta_array(eta_chord, n_chord)
    key = (n_span, n_chord, tuple(np.round(eta_chord, 8)))
    if key in panel_node_cache:
        return panel_node_cache[key]

    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1, 3))
    panel_node_ids = [[] for _ in range(n_span * n_chord)]
    aero_ids = []

    for i_node, X in enumerate(coords):
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            continue
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1e-12)
        if xi < -0.02 or xi > 1.02:
            continue
        z_surf = chord * naca_half_thickness(xi)
        if abs(abs(X[2]) - z_surf) > panel_tol_z:
            continue
        aero_ids.append(i_node)

        eta_s = np.clip(y_val / span, 0.0, 1.0)
        i_span = min(int(eta_s * n_span), n_span - 1)
        j_chord = int(np.argmin(np.abs(eta_chord - xi)))
        x_target = x_le + eta_chord[j_chord] * chord
        if abs(X[0] - x_target) <= panel_tol_x:
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
            z_target = chord * naca_half_thickness(xi_target)
            best_idx = None
            best_d2 = None
            search_ids = aero_ids if aero_ids else range(len(coords))
            for i_node in search_ids:
                X = coords[i_node]
                d2 = (X[0] - x_target) ** 2 + (X[1] - y_target) ** 2 + (abs(X[2]) - z_target) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_idx = i_node
            if best_idx is not None:
                panel_node_ids[panel_idx].append(best_idx)

    panel_node_cache[key] = panel_node_ids
    return panel_node_ids


def update_aero_traction(t_aero, forces, n_span, n_chord, eta_chord):
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
        chord_edges[-1] = min(1.0, eta_chord[-1] + 0.5 * (eta_chord[-1] - eta_chord[-2]))
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
        d_eta_c = max(chord_edges[j_chord + 1] - chord_edges[j_chord], 1e-8)
        d_span = span / n_span
        panel_area_one_side = max(c_mid * d_eta_c * d_span, 1e-10)

        # Incoming fluid force is treated as total panel force; convert to
        # traction and split over top and bottom marked aero surfaces.
        tx = fx / (2.0 * panel_area_one_side)
        ty = fy / (2.0 * panel_area_one_side)
        tz = fz / (2.0 * panel_area_one_side)
        scale = 1.0 / len(ids)
        for i_node in ids:
            vec[dofs_x[i_node]] += scale * tx
            vec[dofs_y[i_node]] += scale * ty
            vec[dofs_z[i_node]] += scale * tz

    vec.apply("insert")


def extract_coupling_nodes(n_span, n_chord, eta_chord):
    eta_chord = as_eta_array(eta_chord, n_chord)
    key = (n_span, n_chord, tuple(np.round(eta_chord, 8)))
    if key in coupling_node_cache:
        return coupling_node_cache[key]

    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1, 3))
    nodes = []

    for i_span in range(n_span):
        y_target = (i_span + 0.5) * span / n_span
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord):
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            z_target = chord * naca_half_thickness(xi_target)

            best_idx = None
            best_d2 = None
            for i_node, X in enumerate(coords):
                d2 = (X[0] - x_target) ** 2 + (X[1] - y_target) ** 2 + (abs(X[2]) - z_target) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_idx = i_node
            nodes.append(coords[best_idx].tolist())

    coupling_node_cache[key] = nodes
    return nodes


def parse_force_payload(data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    if "n_span" in data and "n_chord" in data and "force" in data:
        n_span_in = int(data.get("n_span", 0))
        n_chord_in = int(data.get("n_chord", 0))
        force_raw = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        eta_span_in = as_eta_array(data.get("eta_span"), n_span_in) if n_span_in > 0 else None
        eta_chord_in = as_eta_array(data.get("eta_chord"), n_chord_in) if n_chord_in > 0 else None
        if n_span_in > 0 and n_chord_in > 0:
            forces = resample_forces_to_shape(
                force_raw,
                n_span_out, n_chord_out,
                eta_span_in=eta_span_in, eta_chord_in=eta_chord_in,
                eta_span_out=eta_span_out, eta_chord_out=eta_chord_out
            )
            forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
            return forces, True

    forces_legacy = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
    forces = resample_forces_to_shape(
        forces_legacy,
        n_span_out, n_chord_out,
        eta_span_out=eta_span_out, eta_chord_out=eta_chord_out
    )
    forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
    return forces, False

def get_aero_surface_node_ids():
    '''We are using the logic of "inside" fn defn from AeroSurface class, where the cells which have global coordinates
    taken from tabulate.dof, are checked if they are inside the boundary and then appended to the sorted 
    array called "aero_node_ids", as the internal numbering of the dofs are not in order of the mesh'''
    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1, 3))
    ids = []

    for i_node, X in enumerate(coords):
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            continue
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / max(chord, 1e-12)
        if xi < -0.02 or xi > 1.02:
            continue
        z_surf = chord * naca_half_thickness(xi)
        if abs(abs(X[2]) - z_surf) > panel_tol_z:
            continue
        ids.append(i_node)
    return np.array(sorted(set(ids)), dtype=np.int64), coords

def extract_coupling_node_indices(n_span, n_chord, eta_chord, coords):
    '''Use KD trees for linking the coordinates of the nodes at the surface
    written as outer_surface_coords (global vertices) to the control point locations obtained from eta_coord'''
    eta_chord = as_eta_array(eta_chord, n_chord)
    tree = cKDTree(coords)
    targets = []
    ids = []
    for i_span in range(n_span):
        y_target = (i_span + 0.5) * span / n_span
        chord = chord_at(y_target)
        x_le = x_leading_edge_at(y_target)
        for j_chord in range(n_chord):
            xi_target = eta_chord[j_chord]
            x_target = x_le + xi_target * chord
            z_target = chord * naca_half_thickness(xi_target)
            targets.append([x_target, y_target, z_target])
    _, idx = tree.query(np.asarray(targets), k=1)
    ids = idx.astype(np.int64).tolist()
    return np.asarray(ids, dtype=np.int64)

def build_local_rbf_map(fluid_points, solid_points, epsilon, n_neighbors=32):
    fluid_points = np.asarray(fluid_points, dtype=float)
    solid_points = np.asarray(solid_points, dtype=float)
    eps2 = max(float(epsilon) ** 2, 1e-16)
    n_f = fluid_points.shape[0]
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
    bad = np.where(row_sum[:, 0] <= 1e-16)[0]
    for bi in bad:
        nbr_w[bi, :] = 0.0
        nbr_w[bi, 0] = 1.0
    row_sum = np.maximum(row_sum, 1e-16)
    nbr_w /= row_sum
    return nbr_ids, nbr_w

def map_displacements_to_fluid(u_nodes, nbr_ids, nbr_w):
    n_f, k = nbr_ids.shape
    out = np.zeros((n_f, 3), dtype=float)
    for q in range(k):
        out += nbr_w[:, q:q+1] * u_nodes[nbr_ids[:, q], :]
    return out

def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
    n_f, k = nbr_ids.shape
    out = np.zeros((n_solid_nodes, 3), dtype=float)
    for q in range(k):
        contrib = nbr_w[:, q:q+1] * f_fluid
        np.add.at(out, nbr_ids[:, q], contrib)
    return out

def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
    n_f, k = nbr_ids.shape
    S = np.zeros((n_solid_nodes,), dtype=float)
    for q in range(k):
        np.add.at(S, nbr_ids[:, q], nbr_w[:, q] * A_diag)
    S = np.maximum(S, 1e-14)
    return S

def apply_Tf_operator(Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
    # Paper-style discrete force map:
    #   T_f = S^{-1} T_u^T A
    # with T_u represented by the local RBF weights.
    FaA = Fa * A_diag[:, None]
    rhs = map_forces_to_solid(FaA, n_solid_nodes, nbr_ids, nbr_w)   # T_u^T A Fa
    Fs_coeff = rhs / S_lumped[:, None]                              # S^{-1}(...)
    return Fs_coeff, rhs

def get_nodal_displacements(u_fun, node_ids, dofs_x, dofs_y, dofs_z):
    u_arr = u_fun.vector().get_local()
    out = np.zeros((len(node_ids), 3), dtype=float)
    out[:, 0] = u_arr[dofs_x[node_ids]]
    out[:, 1] = u_arr[dofs_y[node_ids]]
    out[:, 2] = u_arr[dofs_z[node_ids]]
    return out

def apply_nodal_forces_to_residual(res_vec, nodal_forces, node_ids, dofs_x, dofs_y, dofs_z):
    # Residual uses internal - external convention, so mapped external nodal
    # forces are subtracted.
    arr = res_vec.get_local()
    arr[dofs_x[node_ids]] -= nodal_forces[:, 0]
    arr[dofs_y[node_ids]] -= nodal_forces[:, 1]
    arr[dofs_z[node_ids]] -= nodal_forces[:, 2]
    res_vec.set_local(arr)
    res_vec.apply("insert")

def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)

def solve_nonlinear_step(nodal_forces, interface_node_ids, dofs_u_x, dofs_u_y, dofs_u_z,
                         max_iters=40, tol_res_abs=1.0e-6, tol_res_rel=1.0e-6, tol_inc=1.0e-8):
    # Start Newton from previous converged state at current step.
    u.vector()[:] = u_old.vector()
    bc.apply(u.vector())

    du_corr = Function(V)
    converged = False
    res_norm = None
    res0 = None
    for it in range(1, max_iters + 1):
        res_vec = assemble(res)
        if work_conservative_mode and nodal_forces is not None:
            apply_nodal_forces_to_residual(
                res_vec, nodal_forces, interface_node_ids, dofs_u_x, dofs_u_y, dofs_u_z
            )

        bc_hom.apply(res_vec)
        res_norm = res_vec.norm("l2")
        if res0 is None:
            res0 = max(res_norm, 1.0e-16)
        res_target = max(tol_res_abs, tol_res_rel * res0)
        if res_norm < res_target:
            converged = True
            break

        jac_mat = assemble(jac_form)
        bc_hom.apply(jac_mat)

        rhs_newton = res_vec.copy()
        rhs_newton *= -1.0
        solve(jac_mat, du_corr.vector(), rhs_newton, "mumps")

        u.vector().axpy(1.0, du_corr.vector())
        bc.apply(u.vector())

        if du_corr.vector().norm("l2") < tol_inc and res_norm < 10.0 * res_target:
            converged = True
            break

    return converged, it, res_norm if res_norm is not None else 0.0

sig = Function(Vsig, name="sigma") # Stress output field

out_dir = "../results"
os.makedirs(out_dir, exist_ok=True)
xdmf_path = os.path.join(out_dir, "elastodynamics-results-v10-nonlinear-flap.xdmf")
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
print(f"Configured AoA metadata: {aoa_deg} deg")

# Communication discretization (must match fluid panel count)
print("Building interface node sets...")

# Vertices of the cells in the surface
aero_node_ids, outer_surface_coords = get_aero_surface_node_ids()
# Control point of the cells in the surface linked to the nearest node
cp_node_ids = extract_coupling_node_indices(n_span, n_chord, eta_chord_comm, outer_surface_coords)
# 
cp_nodes = outer_surface_coords[cp_node_ids, :]
np.savetxt(os.path.join(out_dir, "coupling_nodes.csv"), cp_nodes, delimiter=",", header="x,y,z", comments="")
print(f"Interface nodes ready: aero={len(aero_node_ids)}, coupling={len(cp_node_ids)}")

# Work-conservative transfer operators:
# fluid panel displacements u_f = G * u_s_nodes
# solid nodal forces      F_s = G^T * F_f
interface_node_ids = aero_node_ids
interface_coords = outer_surface_coords[interface_node_ids, :]
print("Building local RBF transfer map...")
nbr_ids, nbr_w = build_local_rbf_map(cp_nodes, interface_coords, rbf_epsilon, n_neighbors=16)
print(f"Conservative mapping: {cp_nodes.shape[0]} fluid points <- {interface_coords.shape[0]} solid interface nodes")

# Explicit operators used in paper-style formulation:
#   u_f = T_u u_s
#   F_s = T_f F_f, with T_f = S^{-1} T_u^T A
# Here aerodynamic payload is panel-total force, so A = I.
A_diag = np.ones((cp_nodes.shape[0],), dtype=float)
S_lumped = compute_S_lumped(len(interface_node_ids), nbr_ids, nbr_w, A_diag)
print(f"Operator setup: A=I ({len(A_diag)} dofs), S_lumped min/max = {S_lumped.min():.3e}/{S_lumped.max():.3e}")

dofs_u_x = np.asarray(V.sub(0).dofmap().dofs(), dtype=np.int64)
dofs_u_y = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)
dofs_u_z = np.asarray(V.sub(2).dofmap().dofs(), dtype=np.int64)

if work_conservative_mode:
    # Keep variational traction zero; aerodynamic coupling is injected as
    # equivalent nodal loads from conservative transfer.
    t_aero.vector().zero()
    t_aero.vector().apply("insert")

u_cp0 = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
sock.sendall((json.dumps({
    "step": 0,
    "aoa_deg": aoa_deg,
    "n_span": n_span,
    "n_chord": n_chord,
    "eta_span": eta_span_comm.tolist(),
    "eta_chord": eta_chord_comm.tolist(),
    "geometry": u_cp0
}) + "\n").encode())
print("Initial geometry sent.")

time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1,4))
E_damp = 0
force_relax = 1.0 if work_conservative_mode else 0.20
forces_prev = None

for i in range(Nsteps):
    print(f"Solid step {i+1}/{Nsteps}: waiting for force...")

    line = sock_file.readline()
    if line == "":
        raise RuntimeError("Coupling server disconnected while sending force data")
    data = json.loads(line)
    forces, used_2d_force = parse_force_payload(
        data, n_span, n_chord, eta_span_comm, eta_chord_comm
    )
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i+1}")
    if len(forces) != m_panels_comm:
        print(f"Solid step {i+1}/{Nsteps}: force count mismatch (got {len(forces)}, expected {m_panels_comm}), resampling.")
        forces = resample_forces_to_shape(
            forces, n_span, n_chord,
            eta_span_out=eta_span_comm, eta_chord_out=eta_chord_comm
        )
    if i == 0:
        if used_2d_force:
            print(f"Solid: received 2D force payload ({n_span}x{n_chord})")
        else:
            print("Solid: received legacy spanwise force payload and remapped to 2D grid")

    if forces_prev is None:
        forces_eff = forces.copy()
    else:
        #doubtful if this is the way we interpolate forces for generalized alpha method???
        #or what is this then?
        forces_eff = force_relax*forces + (1.0-force_relax)*forces_prev
    forces_prev = forces_eff.copy()

    nodal_forces = None
    Fs_coeff = None

    t = time[i+1]
    ramp = 1.0 if flap_ramp_time <= 0.0 else min(1.0, t / flap_ramp_time)
    flap_theta = ramp * flap_amp * np.sin(2.0 * np.pi * flap_freq_hz * t + flap_phase)
    flap_bc_expr.theta = float(flap_theta)
    if i == 0 or (i + 1) % 20 == 0:
        print(f"Flap state step {i+1}: theta={flap_theta*180.0/np.pi:.3f} deg (ramp={ramp:.3f})")

    if work_conservative_mode:
        Fs_coeff, nodal_forces = apply_Tf_operator(
            forces_eff, len(interface_node_ids), nbr_ids, nbr_w, A_diag, S_lumped
        )
        if not np.isfinite(nodal_forces).all():
            raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i+1}")

        # Discrete work audit on transfer operators.
        u_nodes_prev = get_nodal_displacements(u_old, interface_node_ids, dofs_u_x, dofs_u_y, dofs_u_z)
        u_cp_prev = map_displacements_to_fluid(u_nodes_prev, nbr_ids, nbr_w)
        Wf = float(np.sum(u_cp_prev * (forces_eff * A_diag[:, None])))
        Ws = float(np.sum(u_nodes_prev * (Fs_coeff * S_lumped[:, None])))
        rel_work_err = abs(Wf - Ws) / max(abs(Wf), abs(Ws), 1e-16)
        if i == 0 or (i + 1) % 20 == 0:
            print(f"Work audit step {i+1}: Wf={Wf:.6e}, Ws={Ws:.6e}, rel_err={rel_work_err:.3e}")
    else:
        update_aero_traction(t_aero, forces_eff, n_span, n_chord, eta_chord_comm)

    ok, nit, rnorm = solve_nonlinear_step(
        nodal_forces, interface_node_ids, dofs_u_x, dofs_u_y, dofs_u_z
    )
    if not ok:
        raise RuntimeError(
            f"Nonlinear solve failed to converge at step {i+1}/{Nsteps} (iters={nit}, residual={rnorm:.3e})"
        )
    if i == 0 or (i + 1) % 20 == 0:
        print(f"Nonlinear solver step {i+1}: iters={nit}, residual={rnorm:.3e}")

    update_fields(u, u_old, v_old, a_old)

    xdmf_file.write(u, t)
    local_project(sigma_cauchy(u), Vsig, sig)
    xdmf_file.write(sig, t)

    E_elas = assemble(psi_hyperelastic(u_old) * dx)
    E_kin = assemble(0.5 * rho * inner(v_old, v_old) * dx)
    E_damp += dt_value * assemble(c_form(v_old, v_old))
    E_tot = E_elas + E_kin + E_damp

    energies[i+1,:] = np.array([E_elas,E_kin,E_damp,E_tot])
    u_tip[i+1] = u(0.05, span, 0.0)[2]

    # Send geometry for the next fluid step (not needed after final force).
    if i < Nsteps - 1:
        if work_conservative_mode:
            u_nodes = get_nodal_displacements(u, interface_node_ids, dofs_u_x, dofs_u_y, dofs_u_z)
            u_cp_arr = map_displacements_to_fluid(u_nodes, nbr_ids, nbr_w)
            u_cp = u_cp_arr.tolist()
        else:
            u_cp = []
            for j in range(m_panels_comm):
                x_cp, y_cp, z_cp = cp_nodes[j]
                try:
                    ux, uy, uz = u(x_cp, y_cp, z_cp)
                except RuntimeError:
                    # Fallback to the local mid-surface if exact point lookup fails.
                    try:
                        ux, uy, uz = u(x_cp, y_cp, 0.0)
                    except RuntimeError:
                        print(f"Solid step {i+1}/{Nsteps}: failed displacement lookup at panel {j}, using zeros.")
                        ux, uy, uz = 0.0, 0.0, 0.0
                u_cp.append([float(ux), float(uy), float(uz)])

        msg_geo = json.dumps({
            "step": i + 1,
            "aoa_deg": aoa_deg,
            "n_span": n_span,
            "n_chord": n_chord,
            "eta_span": eta_span_comm.tolist(),
            "eta_chord": eta_chord_comm.tolist(),
            "geometry": u_cp
        })
        sock.sendall((msg_geo + "\n").encode())
        print(f"Solid step {i+1}/{Nsteps}: geometry sent.")

sock_file.close()
sock.close()
print("Solid solver finished.")
print(f"Solid field outputs: {xdmf_path}")

plt.figure()
plt.plot(time,u_tip)
plt.xlabel("Time")
plt.ylabel("Tip displacement (z)")
tip_plot = os.path.join(out_dir, "tip_displacement_v10_nonlinear_flap.png")
plt.savefig(tip_plot, dpi=150)
plt.show()

plt.figure()
plt.plot(time,energies)
plt.legend(("elastic","kinetic","damping","total"))
plt.xlabel("Time")
plt.ylabel("Energy")
energy_plot = os.path.join(out_dir, "energies_v10_nonlinear_flap.png")
plt.savefig(energy_plot, dpi=150)
plt.show()
print(f"Saved plots: {tip_plot}, {energy_plot}")
