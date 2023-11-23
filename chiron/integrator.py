# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from openmm import unit

from .potential import NeuralNetworkPotential
from openmm.app import Topology


class LangevinIntegrator:
    def __init__(
        self,
        potential: NeuralNetworkPotential,
        topology: Topology,
        box_vectors=None,
        progress_bar=False,
    ):
        from .utils import get_list_of_mass

        self.box_vectors = box_vectors
        self.progress_bar = progress_bar
        self.potential = potential

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.mass = get_list_of_mass(topology)

    def run(
        self,
        x0,
        temperature: unit.Quantity,
        n_steps: int = 5_000,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        key=random.PRNGKey(0),
    ):
        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(self.mass.value_in_unit_system(unit.md_unit_system))
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        stepsize_unitless = stepsize.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = collision_rate.value_in_unit_system(
            unit.md_unit_system
        )
        # Initialize velocities
        v0 = sigma_v * random.normal(key, x0.shape)

        # Convert to dimensionless quantities
        a = jnp.exp((-collision_rate_unitless * stepsize_unitless))
        b = jnp.sqrt(1 - jnp.exp(-2 * collision_rate_unitless * stepsize_unitless))

        x = x0.value_in_unit_system(unit.md_unit_system)
        v = v0

        traj = [x]

        for step in tqdm(range(n_steps)) if self.progress_bar else range(n_steps):
            key, subkey = random.split(key)

            # Leapfrog integration
            v += (
                (stepsize_unitless * 0.5)
                * self.potential.compute_force(x)
                / mass_unitless
            )
            x += (stepsize_unitless * 0.5) * v

            if self.box_vectors is not None:
                x = x - self.box_vectors * jnp.floor(x / self.box_vectors)

            v = a * v + b * sigma_v * random.normal(subkey, x.shape)

            x += (stepsize_unitless * 0.5) * v
            v += (
                (stepsize_unitless * 0.5)
                * self.potential.compute_force(x)
                / mass_unitless
            )

            traj.append(x)

        return traj
