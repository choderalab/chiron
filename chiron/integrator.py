# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from openmm import unit

from .potential import NeuralNetworkPotential
from openmm.app import Topology
from typing import Dict, Optional
from loguru import logger as log


class LangevinIntegrator:
    def __init__(
        self,
        potential: NeuralNetworkPotential,
        topology: Topology,
        box_vectors=None,
        progress_bar=False,
    ):
        """
        Initialize the LangevinIntegrator object.

        Parameters
        ----------
        potential : NeuralNetworkPotential
            Object representing the potential energy function.
        topology : Topology
            Object representing the molecular system.
        box_vectors : array_like, optional
            Box vectors for periodic boundary conditions.
        progress_bar : bool, optional
            Flag indicating whether to display a progress bar during integration.
        """
        from .utils import get_list_of_mass

        self.box_vectors = box_vectors
        self.progress_bar = progress_bar
        self.potential = potential
        self.velocities = None

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.mass = get_list_of_mass(topology)

    def set_velocities(self, vel: unit.Quantity):
        self.velocities = vel

    def run(
        self,
        x0,
        temperature: unit.Quantity,
        n_steps: int = 5_000,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        key=random.PRNGKey(0),
    ) -> Dict[str, jnp.array]:
        """
        Run the integrator to perform Langevin dynamics molecular dynamics simulation.

        Parameters
        ----------
        x0 : array_like
            Initial positions of the particles.
        temperature : unit.Quantity
            Temperature of the system.
        n_steps : int, optional
            Number of simulation steps to perform.
        stepsize : unit.Quantity, optional
            Time step size for the integration.
        collision_rate : unit.Quantity, optional
            Collision rate for the Langevin dynamics.
        key : jax.random.PRNGKey
            Random key for generating random numbers.

        Returns
        -------
        list of array_like
            Trajectory of particle positions at each simulation step.
        """

        log.info("Running Langevin dynamics")
        log.info(f"n_steps = {n_steps}")
        log.info(f"stepsize = {stepsize}")
        log.info(f"collision_rate = {collision_rate}")
        log.info(f"temperature = {temperature}")

        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(self.mass.value_in_unit_system(unit.md_unit_system))
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        stepsize_unitless = stepsize.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = collision_rate.value_in_unit_system(
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

        x = x0.value_in_unit_system(unit.md_unit_system)
        v = v0

        traj = [x]
        energy = [self.potential.compute_energy(x)]

        random_noise_v = random.normal(key, (n_steps, x.shape[-1]))
        for step in tqdm(range(n_steps)) if self.progress_bar else range(n_steps):
            # v
            v += (
                (stepsize_unitless * 0.5)
                * self.potential.compute_force(x)
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

            F = self.potential.compute_force(x)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless

            traj.append(x)
            energy.append(self.potential.compute_energy(x))
        return {"traj": traj, "energy": energy}
