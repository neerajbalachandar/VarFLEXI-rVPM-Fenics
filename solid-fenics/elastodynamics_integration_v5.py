from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import socket
import json
import os

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

T = 4.0
Nsteps = 200
dt_value = T/Nsteps
dt = Constant(dt_value)

span = 1.0
root_chord = 0.12
tip_chord = 0.08
thickness_ratio = 0.12
leading_edge_sweep = 0.0

nx, ny, nz = 32, 200, 8
mesh = BoxMesh(Point(0.0, 0.0, -0.5), Point(1.0, span, 0.5), nx, ny, nz)
span_strips = 200
# Must match fluid-side panel count (fluid-rvpm/fluid_explicit_vpm.jl with n=50 -> m=100)
m_panels_comm = 100


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
        - 0.3516 * xi_clip**2
        + 0.2843 * xi_clip**3
        - 0.1015 * xi_clip**4
    )


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

def left(x, on_boundary):
    return near(x[1], 0.) and on_boundary

def right(x, on_boundary):
    return near(x[1], 1.) and on_boundary

facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)

panel_tol_x = 0.75 * (root_chord / nx)
panel_tol_z = 1.25 * (root_chord * thickness_ratio / nz)

class AeroSurface(SubDomain):
    def inside(self, X, on_boundary):
        if not on_boundary:
            return False
        y_val = X[1]
        chord = chord_at(y_val)
        if chord <= 0.0:
            return False
        x_le = x_leading_edge_at(y_val)
        xi = (X[0] - x_le) / chord
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

E = 10000.0
nu = 0.2
mu = Constant(E/(2.0*(1.0+nu)))
lmbda = Constant(E*nu/((1.0+nu)*(1.0-2.0*nu)))
rho = Constant(1.0)
eta_m = Constant(0.5)
eta_k = Constant(0.5)

alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma = Constant(0.5+alpha_f-alpha_m)
beta = Constant((gamma+0.5)**2/4.)

du = TrialFunction(V)
u_ = TestFunction(V)
u = Function(V, name="Displacement")
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

zero = Constant((0.0,0.0,0.0))
bc = DirichletBC(V, zero, left)

#strain tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))


#mass form
def m(u, u_):
    return rho*inner(u, u_)*dx


#elastic stiffness form
def k(u, u_):
    return inner(sigma(u), sym(grad(u_)))*dx


#damping
def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)


#work energy form
def Wext(u_):
    return dot(u_, t_aero) * ds_aero(5)

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

a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

res = m(avg(a_old, a_new, alpha_m), u_) \
    + c(avg(v_old, v_new, alpha_f), u_) \
    + k(avg(u_old, du, alpha_f), u_) \
    - Wext(u_)

a_form = lhs(res)
L_form = rhs(res)

K, _ = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True

strip_node_cache = {}


def get_strip_node_ids(n_strips):
    if n_strips in strip_node_cache:
        return strip_node_cache[n_strips]

    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1, 3))
    panel_node_ids = [[] for _ in range(n_strips)]

    for i, X in enumerate(coords):
        y_val = X[1]
        eta = np.clip(y_val / span, 0.0, 1.0)
        panel_idx = min(int(eta * n_strips), n_strips - 1)
        chord = chord_at(y_val)
        x_le = x_leading_edge_at(y_val)
        x_cp = x_le + 0.75 * chord
        xi = (X[0] - x_le) / max(chord, 1e-12)
        z_surf = chord * naca_half_thickness(xi)
        if abs(X[0] - x_cp) <= panel_tol_x and abs(abs(X[2]) - z_surf) <= panel_tol_z:
            panel_node_ids[panel_idx].append(i)

    strip_node_cache[n_strips] = panel_node_ids
    return panel_node_ids

def update_aero_traction(t_aero, forces):
    vec = t_aero.vector()
    vec.zero()

    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1,3))

    dofs_x = Vt.sub(0).dofmap().dofs()
    dofs_y = Vt.sub(1).dofmap().dofs()
    dofs_z = Vt.sub(2).dofmap().dofs()

    m_panels = len(forces)
    panel_node_ids = get_strip_node_ids(m_panels)

    for panel_idx, ids in enumerate(panel_node_ids):
        if not ids:
            continue
        fx, fy, fz = forces[panel_idx]
        scale = 1.0 / len(ids)
        for i in ids:
            vec[dofs_x[i]] += scale * fx
            vec[dofs_y[i]] += scale * fy
            vec[dofs_z[i]] += scale * fz

    vec.apply("insert")


def extract_three_quarter_chord_nodes(n_panels):
    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1, 3))
    nodes = []
    for j in range(n_panels):
        y_target = (j + 0.5) * span / n_panels
        chord = chord_at(y_target)
        x_target = x_leading_edge_at(y_target) + 0.75 * chord
        z_target = chord * naca_half_thickness(0.75)
        best_idx = None
        best_d2 = None
        for i, X in enumerate(coords):
            d2 = (X[0] - x_target)**2 + (X[1] - y_target)**2 + (abs(X[2]) - z_target)**2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i
        nodes.append(coords[best_idx].tolist())
    return nodes

def resample_forces_to_panels(forces, n_panels):
    forces = np.asarray(forces, dtype=float)
    n_in = len(forces)
    if n_in == n_panels:
        return forces
    if n_in <= 1:
        return np.repeat(forces[:1], n_panels, axis=0)

    out = np.zeros((n_panels, 3), dtype=float)
    for i in range(n_panels):
        s = (i * (n_in - 1)) / max(n_panels - 1, 1)
        i0 = int(np.floor(s))
        i1 = int(np.ceil(s))
        w = s - i0
        out[i, :] = (1.0 - w) * forces[i0, :] + w * forces[i1, :]
    return out

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

sig = Function(Vsig, name="sigma")
#change after this run
out_dir = "solid-fenics/results"
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

# Communication discretization (must match fluid panel count)
cp_nodes = extract_three_quarter_chord_nodes(m_panels_comm)
panel_eta_comm = [(j + 0.5) / m_panels_comm for j in range(m_panels_comm)]
np.savetxt(os.path.join(out_dir, "three_quarter_chord_nodes.csv"), np.array(cp_nodes), delimiter=",", header="x,y,z", comments="")
u_cp0 = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
sock.sendall((json.dumps({"step": 0, "geometry": u_cp0, "eta": panel_eta_comm}) + "\n").encode())
print("Initial geometry sent.")

time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1,4))
E_damp = 0
force_relax = 0.20
forces_prev = None

for i in range(Nsteps):
    print(f"Solid step {i+1}/{Nsteps}: waiting for force...")

    line = sock_file.readline()
    if line == "":
        raise RuntimeError("Coupling server disconnected while sending force data")
    data = json.loads(line)
    #what is force here?? need to take a look at the .json files being written
    forces = np.array(data["force"])
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i+1}")
    if len(forces) != m_panels_comm:
        print(f"Solid step {i+1}/{Nsteps}: force count mismatch (got {len(forces)}, expected {m_panels_comm}), resampling.")
        forces = resample_forces_to_panels(forces, m_panels_comm)

    if forces_prev is None:
        forces_eff = forces.copy()
    else:
        #doubtful if this is the way we interpolate forces for generalized alpha method???
        #or what is this then?
        forces_eff = force_relax*forces + (1.0-force_relax)*forces_prev
    forces_prev = forces_eff.copy()

    update_aero_traction(t_aero, forces_eff)

    rhs_vec = assemble(L_form)
    bc.apply(rhs_vec)
    try:
        solver.solve(K, u.vector(), rhs_vec)
    except RuntimeError as err:
        print(f"Primary linear solve failed at solid step {i+1}/{Nsteps}: {err}")
        print("Retrying with freshly assembled matrix and direct LU.")
        K_retry, _ = assemble_system(a_form, L_form, bc)
        solve(K_retry, u.vector(), rhs_vec, "lu")

    update_fields(u, u_old, v_old, a_old)

    t = time[i+1]

    xdmf_file.write(u, t)
    local_project(sigma(u), Vsig, sig)
    xdmf_file.write(sig, t)

    E_elas = assemble(0.5*k(u_old, u_old))
    E_kin = assemble(0.5*m(v_old, v_old))
    E_damp += dt_value*assemble(c(v_old, v_old))
    E_tot = E_elas + E_kin + E_damp

    energies[i+1,:] = np.array([E_elas,E_kin,E_damp,E_tot])
    u_tip[i+1] = u(0.05,1.0,0.0)[1]

    # Send geometry for the next fluid step (not needed after final force)
    # need to check message passing and all the .json files and verify
    if i < Nsteps - 1:
        m_panels = m_panels_comm
        u_cp = []
        for j in range(m_panels):
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
            "geometry": u_cp,
            "eta": panel_eta_comm
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
plt.ylabel("Tip displacement")
tip_plot = os.path.join(out_dir, "tip_displacement.png")
plt.savefig(tip_plot, dpi=150)
plt.show()

plt.figure()
plt.plot(time,energies)
plt.legend(("elastic","kinetic","damping","total"))
plt.xlabel("Time")
plt.ylabel("Energy")
energy_plot = os.path.join(out_dir, "energies.png")
plt.savefig(energy_plot, dpi=150)
plt.show()
print(f"Saved plots: {tip_plot}, {energy_plot}")
