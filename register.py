from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from example_problems.euler_poisson_example import EulerPoisson
from example_problems.flocking_example import Flocking
from example_problems.euler_poisson_with_drift import EulerPoissonWithDrift
from methods.KiNet import KiNet
from methods.PINN import PINN
from omegaconf import DictConfig

PDE_INSTANCES = {
    '2D-Kinetic-Fokker-Planck'  : KineticFokkerPlanck,
    # '3D-Euler-Poisson'          : EulerPoisson,
    '3D-Euler-Poisson'    : EulerPoissonWithDrift,
    '3D-Flocking'               : Flocking,
}
# Methods
METHODS = {
    'KiNet' : KiNet,
    'PINN'  : PINN,
}

def get_pde_instance(cfg: DictConfig):
    return PDE_INSTANCES[f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"]

def get_method(cfg: DictConfig):
    return METHODS[cfg.solver.name]