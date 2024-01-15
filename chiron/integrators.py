# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from openmm import unit
from .states import SamplerState, ThermodynamicState
from .reporters import LangevinDynamicsReporter
from typing import Optional
from .potential import NeuralNetworkPotential
from .neighbors import PairsBase


class LangevinIntegrator:
    """
    Langevin dynamics integrator for molecular dynamics simulation using the BAOAB splitting scheme [1].

    References:
    [1] Benedict Leimkuhler, Charles Matthews;
        Robust and efficient configurational molecular sampling via Langevin dynamics.
        J. Chem. Phys. 7 May 2013; 138 (17): 174102. https://doi.org/10.1063/1.4802990


    """

    def __init__(
        self,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        save_frequency: int = 100,
        reporter: Optional[LangevinDynamicsReporter] = None,
        save_traj_in_memory: bool = False,
    ) -> None:
        """
        Initialize the LangevinIntegrator object.

        Parameters
        ----------
        stepsize : unit.Quantity, optional
            Time step of integration with units of time. Default is 1.0 * unit.femtoseconds.
        collision_rate : unit.Quantity, optional
            Collision rate for the Langevin dynamics, with units 1/time. Default is 1.0 / unit.picoseconds.
        save_frequency : int, optional
            Frequency of saving the simulation data. Default is 100.
        reporter : SimulationReporter, optional
            Reporter object for saving the simulation data. Default is None.
        save_traj_in_memory: bool
            Flag indicating whether to save the trajectory in memory.
            Default is False. NOTE: Only for debugging purposes.
        """
        from loguru import logger as log

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        log.info(f"stepsize = {stepsize}")
        log.info(f"collision_rate = {collision_rate}")
        log.info(f"save_frequency = {save_frequency}")

        self.stepsize = stepsize
        self.collision_rate = collision_rate
        if reporter:
            log.info(f"Using reporter {reporter} saving to {reporter.file_path}")
            self.reporter = reporter
        self.save_frequency = save_frequency
        self.velocities = None
        self.save_traj_in_memory = save_traj_in_memory
        self.traj = []

    def set_velocities(self, vel: unit.Quantity) -> None:
        """
        Set the initial velocities for the Langevin Integrator.

        Parameters
        ----------
        vel : unit.Quantity
            Velocities to be set for the integrator.
        """
        self.velocities = vel

    def run(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        n_steps: int = 5_000,
        nbr_list: Optional[PairsBase] = None,
        progress_bar=False,
    ):
        """
        Run the integrator to perform Langevin dynamics molecular dynamics simulation.

        Parameters
        ----------
        sampler_state : SamplerState
            The initial state of the simulation, including positions.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system, including temperature and potential.
        n_steps : int, optional
            Number of simulation steps to perform.
        nbr_list : PairBase, optional
            Neighbor list for the system.
        progress_bar : bool, optional
            Flag indicating whether to display a progress bar during integration.

        """
        from .utils import get_list_of_mass
        from tqdm import tqdm
        from loguru import logger as log

        potential = thermodynamic_state.potential

        mass = get_list_of_mass(potential.topology)

        self.box_vectors = sampler_state.box_vectors
        self.progress_bar = progress_bar
        temperature = thermodynamic_state.temperature
        x0 = sampler_state.x0

        log.debug("Running Langevin dynamics")
        log.debug(f"n_steps = {n_steps}")
        log.debug(f"temperature = {temperature}")

        # Initialize the random number generator
        key = sampler_state.random_seed

        # Convert to dimensionless quantities
        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(mass.value_in_unit_system(unit.md_unit_system))[
            :, None
        ]
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        stepsize_unitless = self.stepsize.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = self.collision_rate.value_in_unit_system(
            unit.md_unit_system
        )
        a = jnp.exp((-collision_rate_unitless * stepsize_unitless))
        b = jnp.sqrt(1 - jnp.exp(-2 * collision_rate_unitless * stepsize_unitless))

        # Initialize velocities
        if self.velocities is None:
            v0 = sigma_v * random.normal(key, x0.shape)
        else:
            v0 = self.velocities.value_in_unit_system(unit.md_unit_system)

        x = x0
        v = v0

        if nbr_list is not None:
            nbr_list.build_from_state(sampler_state)

        F = potential.compute_force(x, nbr_list)

        # propagation loop
        for step in tqdm(range(n_steps)) if self.progress_bar else range(n_steps):
            key, subkey = random.split(key)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless
            # r
            x += (stepsize_unitless * 0.5) * v

            if nbr_list is not None:
                x = self._wrap_and_rebuild_neighborlist(x, nbr_list)
            # o
            random_noise_v = random.normal(subkey, x.shape)
            v = (a * v) + (b * sigma_v * random_noise_v)

            x += (stepsize_unitless * 0.5) * v
            if nbr_list is not None:
                x = self._wrap_and_rebuild_neighborlist(x, nbr_list)

            F = potential.compute_force(x, nbr_list)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless

            if step % self.save_frequency == 0:
                if hasattr(self, "reporter") and self.reporter is not None:
                    self._report(x, potential, nbr_list, step)

                if self.save_traj_in_memory:
                    self.traj.append(x)

        log.debug("Finished running Langevin dynamics")
        # save the final state of the simulation in the sampler_state object
        sampler_state.x0 = x
        sampler_state.v0 = v
        # self.reporter.close()

    def _wrap_and_rebuild_neighborlist(self, x: jnp.array, nbr_list: PairsBase):
        """
        Wrap the coordinates and rebuild the neighborlist if necessary.

        Parameters
        ----------
        x: jnp.array
            The coordinates of the particles.
        nbr_list: PairsBsse
            The neighborlist object.
        """

        x = nbr_list.space.wrap(x)
        # check if we need to rebuild the neighborlist after moving the particles
        if nbr_list.check(x):
            nbr_list.build(x, self.box_vectors)
        return x

    def _report(
        self,
        x: jnp.array,
        potential: NeuralNetworkPotential,
        nbr_list: PairsBase,
        step: int,
    ):
        """
        Reports the trajectory, energy, step, and box vectors (if available) to the reporter.

        Parameters
        ----------
            x : jnp.array
                current coordinate set
            potential: NeuralNetworkPotential
                potential used to compute the energy and force
            nbr_list: PairsBase
                The neighbor list
            step: int
                The current time step.

        Returns:
            None
        """
        d = {
            "traj": x,
            "energy": potential.compute_energy(x, nbr_list),
            "step": step,
        }
        if nbr_list is not None:
            d["box_vectors"] = nbr_list.space.box_vectors

        self.reporter.report(d)
