# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from openmm import unit

from openmm.app import Topology
from typing import Dict, Optional
from loguru import logger as log
from .states import SamplerState, ThermodynamicState


class LangevinIntegrator:
    def __init__(
        self,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
    ):
        """
        Initialize the LangevinIntegrator object.

        Parameters
        ----------
        stepsize : unit.Quantity, optional
            Time step size for the integration.
        collision_rate : unit.Quantity, optional
            Collision rate for the Langevin dynamics.
        """

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        log.info(f"stepsize = {stepsize}")
        log.info(f"collision_rate = {collision_rate}")
        self.stepsize = stepsize
        self.collision_rate = collision_rate

    def set_velocities(self, vel: unit.Quantity):
        self.velocities = vel

    def run(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        n_steps: int = 5_000,
        key=random.PRNGKey(0),
        progress_bar=False,
    ) -> Dict[str, jnp.array]:
        """
        Run the integrator to perform Langevin dynamics molecular dynamics simulation.

        Parameters
        ----------
        x0 : array_like
            Initial positions of the particles.
        potential : NeuralNetworkPotential
            Object representing the potential energy function. #NOTE: this might change, maybe rename to system
        box_vectors : array_like, optional
            Box vectors for periodic boundary conditions.
        progress_bar : bool, optional
            Flag indicating whether to display a progress bar during integration.
        temperature : unit.Quantity
            Temperature of the system.
        n_steps : int, optional
            Number of simulation steps to perform.
        key : jax.random.PRNGKey
            Random key for generating random numbers.

        Returns
        -------
        list of array_like
            Trajectory of particle positions at each simulation step.
        """
        from .utils import get_list_of_mass

        potential = thermodynamic_state.potential

        self.mass = get_list_of_mass(potential.topology)

        self.box_vectors = sampler_state.box_vectors
        self.progress_bar = progress_bar
        self.velocities = None
        temperature = thermodynamic_state.temperature
        x0 = sampler_state.x0

        log.info("Running Langevin dynamics")
        log.info(f"n_steps = {n_steps}")
        log.info(f"temperature = {temperature}")

        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(self.mass.value_in_unit_system(unit.md_unit_system))
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        stepsize_unitless = self.stepsize.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = self.collision_rate.value_in_unit_system(
            unit.md_unit_system
        )

        # Initialize velocities
        if self.velocities is None:
            v0 = sigma_v * random.normal(key, x0.shape)
        else:
            v0 = self.velocities.value_in_unit_system(unit.md_unit_system)
        # Convert to dimensionless quantities
        a = jnp.exp((-collision_rate_unitless * stepsize_unitless))
        b = jnp.sqrt(1 - jnp.exp(-2 * collision_rate_unitless * stepsize_unitless))

        x = jnp.array(x0.value_in_unit_system(unit.md_unit_system))
        v = jnp.array(v0)

        traj = [x]
        energy = [potential.compute_energy(x)]

        random_noise_v = random.normal(key, (n_steps, x.shape[-1]))
        for step in tqdm(range(n_steps)) if self.progress_bar else range(n_steps):
            # v
            v += (
                (stepsize_unitless * 0.5)
                * potential.compute_force(x)
                / mass_unitless
            )
            # r
            x += (stepsize_unitless * 0.5) * v

            if self.box_vectors is not None:
                x = x - self.box_vectors * jnp.floor(x / self.box_vectors)
            # o
            v = (a * v) + (b * sigma_v * random_noise_v[step])
            # r
            x += (stepsize_unitless * 0.5) * v

            F = potential.compute_force(x)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless

            traj.append(x)
            energy.append(potential.compute_energy(x))

        return {"traj": traj, "energy": energy}
