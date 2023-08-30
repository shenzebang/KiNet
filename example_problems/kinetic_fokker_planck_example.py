import jax.numpy as jnp
from core.distribution import Gaussian, DistributionKinetic, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import QuadraticPotential

Sigma_x_0 = jnp.diag(jnp.array([1., 1.]))
mu_x_0 = jnp.array([2., 2.])
distribution_x_0 = Gaussian(mu_x_0, Sigma_x_0)

Sigma_v_0 = jnp.diag(jnp.array([1, 1]))
mu_v_0 = jnp.array([0., 0.])
distribution_v_0 = Gaussian(mu_v_0, Sigma_v_0)

Sigma_x_inf = jnp.diag(jnp.array([1., 1.]))
mu_x_inf = jnp.array([0., 0.])
target_potential = QuadraticPotential(mu_x_inf, Sigma_x_inf)

beta = 1.  # friction coefficient

Gamma = jnp.sqrt(4 * beta)


f_CLD = jnp.array([[0., 1.], [-beta, - 4 * beta / Gamma]])

G = jnp.array([[0., 0.], [0., jnp.sqrt(2 * Gamma * beta)]])

f_kron_eye = jnp.kron(f_CLD, jnp.eye(2))
GG_transpose = jnp.matmul(G, jnp.transpose(G))




def eval_Gaussian_Sigma_mu_kinetic(Sigma_x_0, mu_x_0, Sigma_v_0, mu_v_0, time_stamps, tolerance=1e-5):
    # TODO: dependence on the diffusion coefficient
    mu_0 = jnp.concatenate([mu_x_0, mu_v_0], axis=-1)
    Sigma_0 = jnp.diag(jnp.concatenate([jnp.diag(Sigma_x_0), jnp.diag(Sigma_v_0)], axis=-1))
    states_0 = [Sigma_0, mu_0]

    def ode_func(states, t):
        Sigma_t, mu_t = states

        dSigma = jnp.matmul(f_kron_eye, Sigma_t) + jnp.transpose(jnp.matmul(f_kron_eye, Sigma_t)) + jnp.kron(
            GG_transpose, jnp.eye(2))
        dmu = jnp.matmul(f_kron_eye, mu_t)
        return [dSigma, dmu]

    states = odeint(ode_func, states_0, time_stamps, atol=tolerance, rtol=tolerance)
    mus = states[1]
    Sigmas = states[0]

    return mus, Sigmas


# test_time_stamps = jnp.linspace(0, T, num=11)
#
# mus, Sigmas = eval_Gaussian_Sigma_mu_kinetic(Sigma_x_0, mu_x_0, Sigma_v_0, mu_v_0, test_time_stamps)
#
# test_data = (test_time_stamps, mus, Sigmas)


class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, args, rng):
        super().__init__(args, rng)
        self.diffusion_coefficient = jnp.ones([]) * args.diffusion_coefficient
        self.total_evolving_time = jnp.ones([]) * args.total_evolving_time
        self.distribution_0 = DistributionKinetic(distribution_x=distribution_x_0, distribution_v=distribution_v_0)

        self.test_data = self.prepare_test_data()
        self.target_potential = target_potential
        self.beta = beta
        self.Gamma = Gamma
        # domain of interest (2d dimensional box)
        effective_domain_dim = args.domain_dim * 2  # (2d for position and velocity)
        self.mins = args.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = args.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (args.domain_max - args.domain_min) ** (effective_domain_dim)

        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * args.total_evolving_time)
        self.distribution_domain = Uniform(self.mins, self.maxs)

    def prepare_test_data(self):
        test_time_stamps = jnp.linspace(jnp.zeros([]), self.total_evolving_time, num=11)

        mus, Sigmas = eval_Gaussian_Sigma_mu_kinetic(Sigma_x_0, mu_x_0, Sigma_v_0, mu_v_0, test_time_stamps)

        test_data = (test_time_stamps, mus, Sigmas)

        return test_data

    def ground_truth(self, xs: jnp.ndarray):
        _, mus, Sigmas = self.test_data

        if xs.ndim == 3:
            v_v_gaussian_score = jax.vmap(v_gaussian_score, in_axes=[0, 0, 0])
            v_v_gaussian_log_density = jax.vmap(v_gaussian_log_density, in_axes=[0, 0, 0])
        elif xs.ndim == 2:
            v_v_gaussian_score = jax.vmap(v_gaussian_score, in_axes=[None, 0, 0])
            v_v_gaussian_log_density = jax.vmap(v_gaussian_log_density, in_axes=[None, 0, 0])
        else:
            raise NotImplementedError

        scores_true = v_v_gaussian_score(xs, Sigmas, mus)
        log_densities_true = v_v_gaussian_log_density(xs, Sigmas, mus)

        return scores_true, log_densities_true
