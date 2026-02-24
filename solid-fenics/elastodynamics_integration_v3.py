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

x, y, z = 0.1, 1.0, 0.01
nx, ny, nz = 10, 200, 5
mesh = BoxMesh(Point(0., 0., 0.), Point(x, y, z), nx, ny, nz)

def left(x, on_boundary):
    return near(x[1], 0.) and on_boundary

def right(x, on_boundary):
    return near(x[1], 1.) and on_boundary

facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)

x_34 = 0.75*x
hx = x/nx
hz = z/nz

class AeroSurface(SubDomain):
    def inside(self, X, on_boundary):
        return on_boundary and near(X[0], x_34, 0.5*hx) and near(X[2], 0.0, 0.5*hz)

aero_surface = AeroSurface()
aero_surface.mark(facet_markers, 5)

ds_aero = Measure("ds", domain=mesh, subdomain_data=facet_markers)

V = VectorFunctionSpace(mesh, "CG", 1)
Vt = VectorFunctionSpace(mesh, "CG", 1)
Vsig = TensorFunctionSpace(mesh, "DG", 0)

t_aero = Function(Vt, name="AerodynamicTraction")

E = 100000.0
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

def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

def m(u, u_):
    return rho*inner(u, u_)*dx

def k(u, u_):
    return inner(sigma(u), sym(grad(u_)))*dx

def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)

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

def update_aero_traction(t_aero, forces):
    vec = t_aero.vector()
    vec.zero()

    tolx = 0.6*(x/nx)
    tolz = 0.6*(z/nz)

    Vscalar = Vt.sub(0).collapse()
    coords = Vscalar.tabulate_dof_coordinates().reshape((-1,3))

    dofs_x = Vt.sub(0).dofmap().dofs()
    dofs_y = Vt.sub(1).dofmap().dofs()
    dofs_z = Vt.sub(2).dofmap().dofs()

    m_panels = len(forces)

    for i, X in enumerate(coords):
        if abs(X[0]-x_34)<tolx and abs(X[2])<tolz:
            eta = np.clip(X[1],0.0,1.0)
            idx = int(eta*(m_panels-1))
            fx, fy, fz = forces[idx]
            vec[dofs_x[i]] = fx
            vec[dofs_y[i]] = fy
            vec[dofs_z[i]] = fz

    vec.apply("insert")

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
print("Solid connected.")

# Communication discretization (must match fluid panel count)
m_panels_comm = 100
u_cp0 = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
sock.sendall((json.dumps({"step": 0, "geometry": u_cp0}) + "\n").encode())
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
    forces = np.array(data["force"])
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i+1}")

    if forces_prev is None:
        forces_eff = forces.copy()
    else:
        forces_eff = force_relax*forces + (1.0-force_relax)*forces_prev
    forces_prev = forces_eff.copy()

    update_aero_traction(t_aero, forces_eff)

    rhs_vec = assemble(L_form)
    bc.apply(rhs_vec)
    solver.solve(K, u.vector(), rhs_vec)

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
    if i < Nsteps - 1:
        m_panels = len(forces_eff)
        u_cp = []
        denom = max(m_panels - 1, 1)
        for j in range(m_panels):
            eta = j/denom
            ux, uy, uz = u(x_34, eta, 0.0)
            u_cp.append([float(ux), float(uy), float(uz)])

        msg_geo = json.dumps({
            "step": i + 1,
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
