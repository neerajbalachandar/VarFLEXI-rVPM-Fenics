from fenics import *
import numpy as np
from fenicsprecice import Adapter

# Geometry and material properties
H, W = 1.0, 0.1
rho, E, nu = 3000.0, 4000000.0, 0.3
mu = Constant(E / (2.0 * (1.0 + nu)))
lambda_ = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
dim = 2
tol = 1E-14

# 1. Mesh and Function Space
# Ensure the mesh density is sufficient to resolve the structural modes
n_x, n_y = 4, 26
mesh = RectangleMesh(Point(-W / 2, 0), Point(W / 2, H), n_x, n_y)
V = VectorFunctionSpace(mesh, 'P', 2) # Quadratic basis for accuracy 

# 2. Boundary Definitions
# The neumann_boundary must encapsulate the entire "wet" surface interacting with VLM
class CouplingBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] > tol) # Everything except the clamped base

coupling_boundary = CouplingBoundary()
fixed_boundary = AutoSubDomain(lambda x, on: on and x[1] <= tol)

# 3. Variational Forms (Generalized Alpha Method)
u_np1, v, du = Function(V), TestFunction(V), TrialFunction(V)
u_n, v_n, a_n = Function(V), Function(V), Function(V)

# Newmark/Alpha parameters
alpha_m, alpha_f = Constant(0.2), Constant(0.4)
gamma = Constant(0.5 + alpha_f - alpha_m)
beta = Constant((gamma + 0.5)**2 / 4.0)

def epsilon(u): return sym(grad(u))
def sigma(u): return lambda_ * div(u) * Identity(dim) + 2 * mu * epsilon(u)

# Residual form including acceleration and internal stress
# Traction forces are applied later as PointSources
def update_a(u, u_old, v_old, a_old):
    return (u - u_old - dt*v_old) / (beta*dt**2) - (1 - 2*beta)/(2*beta)*a_old

a_new = update_a(du, u_n, v_n, a_n)
res = rho * inner(alpha_m*a_n + (1-alpha_m)*a_new, v) * dx + \
      inner(sigma(alpha_f*u_n + (1-alpha_f)*du), epsilon(v)) * dx

a_form, L_form = lhs(res), rhs(res)
bc = DirichletBC(V, Constant((0, 0)), fixed_boundary)

# 4. preCICE Initialization
# The adapter handles conversion between FEniCS and preCICE structures
precice = Adapter(adapter_config_filename="precice-adapter-config-fsi-s.json")
precice.initialize(coupling_boundary, read_function_space=V, write_object=V, fixed_boundary=fixed_boundary)

t, n = 0.0, 0
fenics_dt = precice.get_max_time_step_size()

while precice.is_coupling_ongoing():
    if precice.requires_writing_checkpoint():
        precice.store_checkpoint((u_n, v_n, a_n), t, n) # Save state for implicit iterations

    dt_val = precice.get_max_time_step_size()
    dt = Constant(dt_val)

    # Read force data from FlowVLM
    # FlowVLM provides discrete forces at control points 
    read_data = precice.read_data(dt_val)
    
    # get_point_sources applies forces conservatively to the structural nodes
    Forces_x, Forces_y = precice.get_point_sources(read_data)

    A_mat, b_vec = assemble_system(a_form, L_form, bc)
    
    # Apply aerodynamic loads to the RHS
    for ps in Forces_x: ps.apply(b_vec)
    for ps in Forces_y: ps.apply(b_vec)

    solve(A_mat, u_np1.vector(), b_vec)

    # Write displacements back to preCICE
    # This will be mapped to VLM vertices or control points depending on config
    precice.write_data(u_np1)
    precice.advance(dt_val)

    if precice.requires_reading_checkpoint():
        uva_cp, t, n = precice.retrieve_checkpoint()
        u_n.assign(uva_cp); v_n.assign(uva_cp[1]); a_n.assign(uva_cp[2])
    else:
        # Update Newmark variables
        a_vec = update_a(u_np1.vector(), u_n.vector(), v_n.vector(), a_n.vector())
        v_n.vector()[:] += dt_val * ((1-float(gamma))*a_n.vector() + float(gamma)*a_vec)
        a_n.vector()[:] = a_vec
        u_n.assign(u_np1)
        t += dt_val
        n += 1

precice.finalize()