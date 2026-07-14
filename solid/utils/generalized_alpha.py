from dolfin import *


class GeneralizedAlphaIntegrator:
    """Generalized-alpha time integration method, credits to FEniCs 
        for their open source elastodynamics integration scheme implementation."""

    def __init__(self, dt, alpha_m, alpha_f):
        self.dt = dt
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = Constant(0.5 + self.alpha_f - self.alpha_m)
        self.beta = Constant((self.gamma + 0.5) ** 2 / 4.0)

    def average(self, x_old, x_new, alpha):
        return alpha * x_old + (1.0 - alpha) * x_new

    def update_a(self, q_new, q_prev, v_prev, a_prev, ufl=True):
        dt_ = self.dt if ufl else float(self.dt)
        beta_ = self.beta if ufl else float(self.beta)
        return (q_new - q_prev - dt_ * v_prev) / beta_ / dt_ ** 2 - (
            1.0 - 2.0 * beta_
        ) / (2.0 * beta_) * a_prev

    def update_v(self, a_new, q_prev, v_prev, a_prev, ufl=True):
        dt_ = self.dt if ufl else float(self.dt)
        gamma_ = self.gamma if ufl else float(self.gamma)
        return v_prev + dt_ * ((1.0 - gamma_) * a_prev + gamma_ * a_new)

    def update_fields(self, q_fun, q_prev, v_prev, a_prev):
        q_vec = q_fun.vector()
        q0_vec = q_prev.vector()
        v0_vec = v_prev.vector()
        a0_vec = a_prev.vector()

        a_vec = self.update_a(q_vec, q0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = self.update_v(a_vec, q0_vec, v0_vec, a0_vec, ufl=False)

        v_prev.vector()[:] = v_vec
        a_prev.vector()[:] = a_vec
        q_prev.vector()[:] = q_vec


__all__ = ["GeneralizedAlphaIntegrator"]
