import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density, gaussian_log_density
from core.potential import QuadraticPotential, GMMPotential

def _Kinetic_OU_process(t, configuration, beta, Gamma):
    assert t.ndim == 0
    domain_dim = configuration["mu_x_0"].shape[0]
    Bt = beta * t
    mu_x_t = (2 * Bt / Gamma * configuration["mu_x_0"] + 4 * Bt / Gamma ** 2 * configuration["mu_v_0"] + configuration["mu_x_0"]) * jnp.exp(-2 * Bt / Gamma)
    mu_v_t = (-Bt * configuration["mu_x_0"] - 2 * Bt / Gamma * configuration["mu_v_0"] + configuration["mu_v_0"]) * jnp.exp(-2 * Bt / Gamma)
    mu_t = jnp.concatenate([mu_x_t, mu_v_t], axis=-1)

    Sigma_xx_t_scale = configuration["Sigma_x_0_scale"] + jnp.exp(4 * Bt / Gamma) - 1 + 4 * Bt / Gamma * (configuration["Sigma_x_0_scale"] - 1) + 4 * Bt**2 / Gamma**2 * (configuration["Sigma_x_0_scale"] - 2) + 16 * Bt **2 / Gamma**4 * configuration["Sigma_v_0_scale"]
    Sigma_xv_t_scale = - Bt * configuration["Sigma_x_0_scale"] + 4 * Bt / Gamma**2 * configuration["Sigma_v_0_scale"] - 2 * Bt**2 / Gamma * (configuration["Sigma_x_0_scale"] - 2) - 8 * Bt**2 / Gamma**3 * configuration["Sigma_v_0_scale"]
    Sigma_vv_t_scale = Gamma**2/4 * (jnp.exp(4*Bt/Gamma) - 1) + Bt * Gamma + configuration["Sigma_v_0_scale"] * (1 + 4 * Bt**2 / Gamma**2 - 4 * Bt / Gamma) + Bt**2 * (configuration["Sigma_x_0_scale"] - 2)
    Sgima_t_scale = jnp.array([[Sigma_xx_t_scale, Sigma_xv_t_scale], [Sigma_xv_t_scale, Sigma_vv_t_scale]]) * jnp.exp(-4*Bt/Gamma)
    Sigma_t = jnp.kron(Sgima_t_scale, jnp.eye(domain_dim))

    return mu_t, Sigma_t

_Kinetic_OU_process_vmap_t = jax.vmap(_Kinetic_OU_process, in_axes=[0, None, None, None])

def Kinetic_OU_process(t: jnp.ndarray, configuration, beta, Gamma):
    if t.ndim == 0 or (t.ndim == 1 and len(t) == 1): # not batched
        if t.ndim == 1:
            t = t[0]
        return _Kinetic_OU_process(t, configuration, beta, Gamma)
    elif t.ndim == 1 and len(t) != 1:
        return _Kinetic_OU_process_vmap_t(t, configuration, beta, Gamma)
    else:
        raise ValueError("t should be either scalar (0D) or 1D array!")



# def eval_Gaussian_Sigma_mu_kinetic(configuration, time_stamps, beta, Gamma, tolerance=1e-5):
#     # domain_dim = configuration["mu_x_0"].shape[0]

#     # f_CLD = jnp.array([[0., 1.], [-beta, - 4 * beta / Gamma]])
#     # f_kron_eye = jnp.kron(f_CLD, jnp.eye(domain_dim))
#     # G = jnp.array([[0., 0.], [0., jnp.sqrt(2 * Gamma * beta)]])
#     # GG_transpose = G @ jnp.transpose(G)

#     # states_0 = {
#     #     "Sigma": jnp.diag(
#     #         jnp.concatenate(
#     #         [jnp.diag(configuration["Sigma_x_0"]), jnp.diag(configuration["Sigma_v_0"])],
#     #         axis=-1
#     #         )
#     #     ),
#     #     "mu": jnp.concatenate([configuration["mu_x_0"], configuration["mu_v_0"]], axis=-1),
#     # }
#     #
#     # def ode_func(states, t):
#     #     return {
#     #         "Sigma": f_kron_eye @ states["Sigma"] + jnp.transpose(f_kron_eye @ states["Sigma"]) + jnp.kron(
#     #         GG_transpose, jnp.eye(domain_dim)),
#     #         "mu": jnp.matmul(f_kron_eye, states["mu"])
#     #     }
#     #
#     # states = odeint(ode_func, states_0, time_stamps, atol=tolerance, rtol=tolerance)

#     # Check the correctness of the closed form solution
#     mus_closed_form, Sigmas_closed_form = v_Gaussian_Sigma_mu_kinetic_close_form(time_stamps, configuration, beta, Gamma)
#     # print(jnp.mean(jnp.sum((mus_closed_form - states["mu"]) ** 2, axis=-1)))
#     # print(jnp.mean(jnp.sum((Sigmas_closed_form - states["Sigma"]) ** 2, axis=(-1, -2,))))

#     # return states["mu"], states["Sigma"]
#     return mus_closed_form, Sigmas_closed_form


def initialize_configuration(domain_dim: int, beta):
    Sigma_x_0_scale = 1.
    Sigma_v_0_scale = 1.
    return {
        "Sigma_x_0_scale": Sigma_x_0_scale,
        "Sigma_x_0": jnp.eye(domain_dim) * Sigma_x_0_scale,
        # "mu_x_0": jnp.ones(domain_dim) * 2.,
        "mu_x_0": jnp.ones(domain_dim) * 0.,
        "Sigma_v_0_scale": Sigma_v_0_scale,
        "Sigma_v_0": jnp.eye(domain_dim) * Sigma_v_0_scale,
        "mu_v_0": jnp.zeros(domain_dim),
        # for OU
        "Sigma_x_inf": jnp.eye(domain_dim) / beta,
        "mu_x_inf": jnp.zeros(domain_dim),
        # for GMM
        "cov_GMM": jnp.ones([]) * 1, 
        "mus_GMM": jnp.stack([jnp.array([-2., -2.]), jnp.array([-2., 2.]), jnp.array([2., -2.]), jnp.array([2., 2.])], axis=0),
    }

def get_distribution_t(t, configuration, beta, Gamma):
    assert t.ndim == 0
    mu_t, Sigma_t = Kinetic_OU_process(t, configuration, beta, Gamma)
    return Gaussian(mu_t, Sigma_t)
    # distribution_x_0 = Gaussian(configuration["mu_x_0"], configuration["Sigma_x_0"])
    # distribution_v_0 = Gaussian(configuration["mu_v_0"], configuration["Sigma_v_0"])
    # return DistributionKinetic(distribution_x=distribution_x_0, distribution_v=distribution_v_0)

def get_potential(potential_type, configuration):
    if potential_type == "OU":
        return QuadraticPotential(configuration["mu_x_inf"], configuration["Sigma_x_inf"])
    elif potential_type == "GMM":
        return GMMPotential(mus=configuration["mus_GMM"], cov=configuration["cov_GMM"])
    else:
        raise ValueError("Unknown potential")

def prepare_test_data(configuration, total_evolving_time, beta, Gamma):
    test_time_stamps = jnp.linspace(jnp.zeros([]), total_evolving_time, num=11)

    mus, Sigmas = Kinetic_OU_process(test_time_stamps, configuration, beta, Gamma)

    test_data = (test_time_stamps, mus, Sigmas)

    return test_data

class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)

        # Configurations that lead to an analytical solution
        # To ensure that dx = v dt, we have Gamma == jnp.sqrt(4 * beta) and Gamma * beta == diffusion_coefficient
        self.beta = (self.diffusion_coefficient / 2) ** (2 / 3)
        self.Gamma = 2 * jnp.sqrt(self.beta)
        self.initial_configuration = initialize_configuration(cfg.pde_instance.domain_dim, self.beta)
        self.target_potential = get_potential(self.cfg.pde_instance.potential, self.initial_configuration)

        # Analytical solution
        distribution_0 = get_distribution_t(jnp.zeros([]), self.initial_configuration, self.beta, self.Gamma)
        self.u_0 = distribution_0.density

        def u_t(t: jnp.ndarray, x: jnp.ndarray):
            assert t.ndim == 0
            mu_t, cov_t = Kinetic_OU_process(t, self.initial_configuration, self.beta, self.Gamma)
            log_u_t = gaussian_log_density(x, cov_t, mu_t)
            return jnp.exp(log_u_t)
        
        self.u_t = u_t

        # Distributions for KiNet
        self.distribution_0 = distribution_0

        # Distributions for PINN
        effective_domain_dim = cfg.pde_instance.domain_dim * 2  # (2d for position and velocity)
        self.mins = cfg.pde_instance.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = cfg.pde_instance.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (cfg.pde_instance.domain_min - cfg.pde_instance.domain_max) ** (effective_domain_dim)

        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * cfg.pde_instance.total_evolving_time)
        self.distribution_domain = Uniform(self.mins, self.maxs)
        self.log_prob_0 = distribution_0.logdensity

        # Test data
        if self.cfg.pde_instance.perform_test:
            self.test_data = prepare_test_data(self.initial_configuration, self.total_evolving_time, self.beta, self.Gamma)

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        # TODO: revise the implementation of the ground truth fn for FPE to accept any testing time.
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

    def forward_fn_to_dynamics(self, forward_fn):
        def dynamics(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = - self.Gamma * self.beta * forward_fn(t, z) - self.target_potential.gradient(x) - 4 * self.beta / self.Gamma * v
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        return dynamics
