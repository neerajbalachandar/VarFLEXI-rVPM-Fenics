from dolfin import *
import numpy as np

# from FINAL.solid.elastodynamics_reisner_span_for_validation import left

from .generalized_alpha import GeneralizedAlphaIntegrator


class ReissnerMindlinModel:
    """Mixed Reissner-Mindlin plate model with generalized-alpha time stepping.
       Credits to fenics_shells for the original implementation of the Reissner-Mindlin model in FEniCS."""

    def __init__(
        self,
        domain,
        E,
        nu,
        rho_s,
        eta_m,
        eta_k,
        kappa_shear,
        dt,
        alpha_m=0.10,
        alpha_f=0.20,
    ):
        self.domain = domain
        self.E = float(E)
        self.nu = float(nu)
        self.rho_s = float(rho_s)
        self.rho = Constant(self.rho_s)
        self.eta_m = Constant(float(eta_m))
        self.eta_k = Constant(float(eta_k))
        self.kappa_shear = Constant(float(kappa_shear))
        self.h = domain.h
        self.dt = dt
        self.integrator = GeneralizedAlphaIntegrator(dt, Constant(alpha_m), Constant(alpha_f))
        self.gamma = self.integrator.gamma
        self.beta = self.integrator.beta
        self.I2 = Identity(2)

        self._build_spaces()
        self._build_state()
        self._build_bcs()
        # self._build_forms() # Commented and check out what is the fault here, added the below line to correct as it was similar. where is the defn?
        self.build_residual_forms()

    def _build_spaces(self):
        mesh = self.domain.mesh
        U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
        W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
        T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
        mixed_element = MixedElement([U_el, W_el, T_el])
        self.V = FunctionSpace(mesh, mixed_element)
        self.Vt = VectorFunctionSpace(mesh, "CG", 1, dim=3)
        self.Vsig = TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))
        self.t_aero = Function(self.Vt, name="AerodynamicTraction")

    def _build_state(self):
        self.dq_trial = TrialFunction(self.V)
        self.q_test = TestFunction(self.V)
        self.q = Function(self.V, name="PlateState")
        self.q_old = Function(self.V)
        self.v_old = Function(self.V)
        self.a_old = Function(self.V)
        self.dq_newton = Function(self.V)
        self.linear_solver = LUSolver("mumps")
        self.sig = Function(self.Vsig, name="MembraneStress")

    def _build_bcs(self):
        u_zero_2d = Constant((0.0, 0.0))

        # print("========== BC DEBUG ==========")
        # print("V              :", self.V)
        # print("V.sub(0)       :", self.V.sub(0))
        # print("V.sub(1)       :", self.V.sub(1))
        # print("V.sub(2)       :", self.V.sub(2))
        # print("u_zero_2d      :", u_zero_2d)
        # print("is_left_boundary:", self.domain.is_left_boundary)
        # print("type:", type(self.domain.is_left_boundary))
        # print("==============================")
        
        
        # self.bc_u = DirichletBC(self.V.sub(0), u_zero_2d, self.domain.is_left_boundary)     
        # self.bc_w = DirichletBC(self.V.sub(1), Constant(0.0), self.domain.is_left_boundary)
        # self.bc_theta = DirichletBC(self.V.sub(2), u_zero_2d, self.domain.is_left_boundary)

# Why is left BC import failing previously?
        left = CompiledSubDomain("near(x[1], 0.0) && on_boundary")

        self.bc_u = DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), left)
        self.bc_w = DirichletBC(self.V.sub(1), Constant(0.0), left)
        self.bc_theta = DirichletBC(self.V.sub(2), Constant((0.0, 0.0)), left)

        self.bcs = [self.bc_u, self.bc_w, self.bc_theta]

    def split_state(self, q_fun):
        if isinstance(q_fun, tuple):
            return q_fun
        return split(q_fun)

    def membrane_strain(self, u_mem):
        return sym(grad(u_mem))

    def curvature(self, theta):
        return sym(grad(theta))

    def shear_strain(self, theta, w):
        return grad(w) - theta

    def membrane_stress(self, u_mem):
        eps = self.membrane_strain(u_mem)
        coeff = self.E * self.h / (1.0 - self.nu ** 2)
        return coeff * ((1.0 - self.nu) * eps + self.nu * tr(eps) * self.I2)

    def bending_moment(self, theta):
        kap = self.curvature(theta)
        coeff = self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))
        return coeff * ((1.0 - self.nu) * kap + self.nu * tr(kap) * self.I2)

    def displacement_3d(self, q_fun):
        u_mem, w, _theta = self.split_state(q_fun)
        return as_vector((u_mem[0], u_mem[1], w))

    def mass_form(self, q_trial, q_test):
        if isinstance(q_trial, tuple):
            u_t, w_t, theta_t = q_trial
        else:
            u_t, w_t, theta_t = split(q_trial)

        if isinstance(q_test, tuple):
            u_x, w_x, theta_x = q_test
        else:
            u_x, w_x, theta_x = split(q_test)

        inertia_rot = self.rho * self.h ** 3 / 12.0
        return self.rho * self.h * (inner(u_t, u_x) + w_t * w_x) * dx + inertia_rot * inner(
            theta_t, theta_x
        ) * dx

    def stiffness_form(self, q_trial, q_test):
        if isinstance(q_trial, tuple):
            u_t, w_t, theta_t = q_trial
        else:
            u_t, w_t, theta_t = split(q_trial)

        if isinstance(q_test, tuple):
            u_x, w_x, theta_x = q_test
        else:
            u_x, w_x, theta_x = split(q_test)

        eps_x = self.membrane_strain(u_x)
        kap_x = self.curvature(theta_x)
        gam_t = self.shear_strain(theta_t, w_t)
        gam_x = self.shear_strain(theta_x, w_x)

        N_t = self.membrane_stress(u_t)
        M_t = self.bending_moment(theta_t)
        G_shear = Constant(self.E / (2.0 * (1.0 + self.nu)))
        K_shear = self.kappa_shear * G_shear * self.h
        return (
            inner(N_t, eps_x) * dx
            + inner(M_t, kap_x) * dx
            + K_shear * inner(gam_t, gam_x) * dx
        )

    def damping_form(self, q_trial, q_test):
        return self.eta_m * self.mass_form(q_trial, q_test) + self.eta_k * self.stiffness_form(
            q_trial, q_test
        )

    def external_work(self, q_test):
        v_disp = self.displacement_3d(q_test)
        return dot(v_disp, self.t_aero) * self.domain.ds_aero(1)

    def build_residual_forms(self):
        q_u, q_w, q_th = split(self.q)
        qo_u, qo_w, qo_th = split(self.q_old)
        vo_u, vo_w, vo_th = split(self.v_old)
        ao_u, ao_w, ao_th = split(self.a_old)

        a_u_new = self.integrator.update_a(q_u, qo_u, vo_u, ao_u, ufl=True)
        a_w_new = self.integrator.update_a(q_w, qo_w, vo_w, ao_w, ufl=True)
        a_th_new = self.integrator.update_a(q_th, qo_th, vo_th, ao_th, ufl=True)

        v_u_new = self.integrator.update_v(a_u_new, qo_u, vo_u, ao_u, ufl=True)
        v_w_new = self.integrator.update_v(a_w_new, qo_w, vo_w, ao_w, ufl=True)
        v_th_new = self.integrator.update_v(a_th_new, qo_th, vo_th, ao_th, ufl=True)

        a_alpha = (
            self.integrator.average(ao_u, a_u_new, self.integrator.alpha_m),
            self.integrator.average(ao_w, a_w_new, self.integrator.alpha_m),
            self.integrator.average(ao_th, a_th_new, self.integrator.alpha_m),
        )
        v_alpha = (
            self.integrator.average(vo_u, v_u_new, self.integrator.alpha_f),
            self.integrator.average(vo_w, v_w_new, self.integrator.alpha_f),
            self.integrator.average(vo_th, v_th_new, self.integrator.alpha_f),
        )
        q_alpha = (
            self.integrator.average(qo_u, q_u, self.integrator.alpha_f),
            self.integrator.average(qo_w, q_w, self.integrator.alpha_f),
            self.integrator.average(qo_th, q_th, self.integrator.alpha_f),
        )

        self.residual = (
            self.mass_form(a_alpha, self.q_test)
            + self.damping_form(v_alpha, self.q_test)
            + self.stiffness_form(q_alpha, self.q_test)
        )
        self.jacobian = derivative(self.residual, self.q, self.dq_trial)
        return self.residual, self.jacobian

    def update_fields(self):
        self.integrator.update_fields(self.q, self.q_old, self.v_old, self.a_old)

    def compute_strain_energy(self):
        u_mem_t, w_t, theta_t = split(self.q_old)
        return assemble(
            0.5 * inner(self.membrane_stress(u_mem_t), self.membrane_strain(u_mem_t)) * dx
            + 0.5 * inner(self.bending_moment(theta_t), self.curvature(theta_t)) * dx
            + 0.5
            * self.kappa_shear
            * (self.E / (2.0 * (1.0 + self.nu)))
            * self.h
            * inner(self.shear_strain(theta_t, w_t), self.shear_strain(theta_t, w_t))
            * dx
        )

    def compute_kinetic_energy(self):
        return 0.5 * assemble(self.mass_form(self.v_old, self.v_old))

    def compute_damping_increment(self, dt_value):
        return float(dt_value) * assemble(self.damping_form(self.v_old, self.v_old))

    def extract_dof_maps(self):
        dofs_u_x = np.asarray(self.V.sub(0).sub(0).dofmap().dofs(), dtype=np.int64)
        dofs_u_y = np.asarray(self.V.sub(0).sub(1).dofmap().dofs(), dtype=np.int64)
        dofs_w = np.asarray(self.V.sub(1).dofmap().dofs(), dtype=np.int64)
        return dofs_u_x, dofs_u_y, dofs_w


__all__ = ["ReissnerMindlinModel"]
