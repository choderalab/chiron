# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from openmm import unit
from .states import SamplerState, ThermodynamicState
from typing import Dict
from loguru import logger as log
from .reporters import SimulationReporter
from typing import Optional


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
        reporter: Optional[SimulationReporter] = None,
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
        """

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        log.info(f"stepsize = {stepsize}")
        log.info(f"collision_rate = {collision_rate}")
        log.info(f"save_frequency = {save_frequency}")

        self.stepsize = stepsize
        self.collision_rate = collision_rate
        if reporter is not None:
            log.info(f"Using reporter {reporter} saving to {reporter.filename}")
            self.reporter = reporter
        self.save_frequency = save_frequency

        self.velocities = None

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
        key=random.PRNGKey(0),
        nbr_list=None,
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
        key : jax.random.PRNGKey, optional
            Random key for generating random numbers.
        nbr_list : NeighborListNsqrd, optional
            Neighbor list for the system.
        progress_bar : bool, optional
            Flag indicating whether to display a progress bar during integration.

        """
        from .utils import get_list_of_mass

        potential = thermodynamic_state.potential

        mass = get_list_of_mass(potential.topology)

        self.box_vectors = sampler_state.box_vectors
        self.progress_bar = progress_bar
        temperature = thermodynamic_state.temperature
        x0 = sampler_state.x0

        log.info("Running Langevin dynamics")
        log.info(f"n_steps = {n_steps}")
        log.info(f"temperature = {temperature}")
        log.info(f"Using seed: {key}")
        self.key = key
        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(mass.value_in_unit_system(unit.md_unit_system))[
            :, None
        ]
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        stepsize_unitless = self.stepsize.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = self.collision_rate.value_in_unit_system(
            unit.md_unit_system
        )

        # Initialize velocities
        if self.velocities is None:
            v0 = sigma_v * random.normal(self.key, x0.shape)
        else:
            v0 = self.velocities.value_in_unit_system(unit.md_unit_system)
        # Convert to dimensionless quantities
        a = jnp.exp((-collision_rate_unitless * stepsize_unitless))
        b = jnp.sqrt(1 - jnp.exp(-2 * collision_rate_unitless * stepsize_unitless))

        x = x0
        v = v0
        if nbr_list is not None:
            nbr_list.build_from_state(sampler_state)

        F = potential.compute_force(x, nbr_list)
        for step in tqdm(range(n_steps)) if self.progress_bar else range(n_steps):
            self.key, subkey = random.split(self.key)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless
            # r
            x += (stepsize_unitless * 0.5) * v

            if nbr_list is not None:
                x = nbr_list.space.wrap(x)
                # check if we need to rebuild the neighborlist after moving the particles
                if nbr_list.check(x):
                    nbr_list.build(x, self.box_vectors)
            # o
            random_noise_v = random.normal(subkey, x.shape)
            v = (a * v) + (b * sigma_v * random_noise_v)

            x += (stepsize_unitless * 0.5) * v
            if nbr_list is not None:
                x = nbr_list.space.wrap(x)
                # check if we need to rebuild the neighborlist after moving the particles
                if nbr_list.check(x):
                    nbr_list.build(x, self.box_vectors)

            F = potential.compute_force(x, nbr_list)
            # v
            v += (stepsize_unitless * 0.5) * F / mass_unitless

            if step % self.save_frequency == 0:
                # log.debug(f"Saving at step {step}")
                if self.reporter is not None:
                    d = {
                        "traj": x,
                        "energy": potential.compute_energy(x, nbr_list),
                        "step": step,
                    }
                    if nbr_list is not None:
                        d["box_vectors"] = nbr_list.space.box_vectors

                    # log.debug(d)
                    self.reporter.report(d)
        sampler_state.x0 = x
        log.debug("Finished running Langevin dynamics")
        # self.reporter.close()
