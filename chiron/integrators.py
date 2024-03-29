# This file contains the integrator class for the Langevin dynamics simulation

import jax.numpy as jnp
from jax import random
from openmm import unit
from .states import SamplerState, ThermodynamicState
from .reporters import LangevinDynamicsReporter
from typing import Optional, Tuple
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
        timestep=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        refresh_velocities: bool = False,
        report_interval: int = 100,
        reporter: Optional[LangevinDynamicsReporter] = None,
        save_traj_in_memory: bool = False,
    ) -> None:
        """
        Initialize the LangevinIntegrator object.

        Parameters
        ----------
        timestep : unit.Quantity, optional
            Time step of integration with units of time. Default is 1.0 * unit.femtoseconds.
        collision_rate : unit.Quantity, optional
            Collision rate for the Langevin dynamics, with units 1/time. Default is 1.0 / unit.picoseconds.
        refresh_velocities : bool, optional
            Flag indicating whether to reinitialize the velocities each time the run function is called. Default is False.
        report_interval : int, optional
            Interval between saving the simulation data. Default is 100.
        reporter : SimulationReporter, optional
            Reporter object for saving the simulation data. Default is None.
        save_traj_in_memory: bool
            Flag indicating whether to save the trajectory in memory.
            Default is False. NOTE: Only for debugging purposes.
        """
        from loguru import logger as log

        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        log.info(f"timestep = {timestep}")
        log.info(f"collision_rate = {collision_rate}")
        log.info(f"report_interval = {report_interval}")

        self.timestep = timestep
        self.collision_rate = collision_rate
        if reporter:
            log.info(
                f"Using reporter {reporter} saving trajectory to {reporter.xtc_file_path}"
            )
            log.info(f"and logging to {reporter.log_file_path}")
            self.reporter = reporter
        self.report_interval = report_interval
        self.velocities = None
        self.save_traj_in_memory = save_traj_in_memory
        self.traj = []
        self.refresh_velocities = refresh_velocities
        self._move_iteration = 0

    def run(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        number_of_steps: int = 5_000,
        nbr_list: Optional[PairsBase] = None,
        progress_bar=False,
    ) -> Tuple[SamplerState, PairsBase]:
        """
        Run the integrator to perform Langevin dynamics molecular dynamics simulation.

        Parameters
        ----------
        sampler_state : SamplerState
            The initial state of the simulation, including positions.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system, including temperature and potential.
        number_of_steps : int, optional
            Number of simulation steps to perform.
        nbr_list : PairBase, optional
            Neighbor list for the system.
        progress_bar : bool, optional
            Flag indicating whether to display a progress bar during integration.

        Returns
        -------
        sampler_state : SamplerState
            The final state of the simulation, including positions, velocities, and current PRNG key.
        nbr_list : PairBase
            The neighbor list for the final state of the simulation. If the NeighborList object is None, the function returns None.
        """
        from .utils import get_list_of_mass
        from tqdm import tqdm
        from loguru import logger as log

        potential = thermodynamic_state.potential

        mass = get_list_of_mass(potential.topology)

        self.box_vectors = sampler_state.box_vectors
        self.progress_bar = progress_bar
        temperature = thermodynamic_state.temperature
        x0 = sampler_state.positions

        log.debug("Running Langevin dynamics")
        log.debug(f"number_of_steps = {number_of_steps}")
        log.debug(f"temperature = {temperature}")

        # Initialize the random number generator
        key = sampler_state.new_PRNG_key

        # Convert to dimensionless quantities
        kbT_unitless = (self.kB * temperature).value_in_unit_system(unit.md_unit_system)
        mass_unitless = jnp.array(mass.value_in_unit_system(unit.md_unit_system))[
            :, None
        ]
        sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)
        timestep_unitless = self.timestep.value_in_unit_system(unit.md_unit_system)
        collision_rate_unitless = self.collision_rate.value_in_unit_system(
            unit.md_unit_system
        )
        a = jnp.exp((-collision_rate_unitless * timestep_unitless))
        b = jnp.sqrt(1 - jnp.exp(-2 * collision_rate_unitless * timestep_unitless))

        # Initialize velocities
        if self.refresh_velocities:
            # v0 = sigma_v * random.normal(key, positions.shape)
            from .utils import initialize_velocities

            sampler_state.velocities = initialize_velocities(
                temperature, potential.topology, key
            )

        elif sampler_state._velocities is None:
            # v0 = sigma_v * random.normal(key, positions.shape)
            from .utils import initialize_velocities

            sampler_state.velocities = initialize_velocities(
                temperature, potential.topology, key
            )
        elif sampler_state._velocities.shape[0] != sampler_state.positions.shape[0]:
            from .utils import initialize_velocities

            sampler_state.velocities = initialize_velocities(
                temperature, potential.topology, key
            )

        # extract the velocities from the sampler state
        v0 = sampler_state.velocities

        x = x0
        v = v0

        if nbr_list is not None:
            nbr_list.build_from_state(sampler_state)

        F = potential.compute_force(x, nbr_list)

        # propagation loop
        for step in (
            tqdm(range(number_of_steps))
            if self.progress_bar
            else range(number_of_steps)
        ):
            key, subkey = random.split(key)
            # v
            v += (timestep_unitless * 0.5) * F / mass_unitless
            # r
            x += (timestep_unitless * 0.5) * v

            random_noise_v = random.normal(subkey, x.shape)
            v = (a * v) + (b * sigma_v * random_noise_v)

            x += (timestep_unitless * 0.5) * v

            if nbr_list is not None:
                x, nbr_list = self._wrap_and_rebuild_neighborlist(x, nbr_list)

            F = potential.compute_force(x, nbr_list)
            # v
            v += (timestep_unitless * 0.5) * F / mass_unitless

            elapsed_step = step + self._move_iteration * number_of_steps
            if (elapsed_step) % self.report_interval == 0:
                if hasattr(self, "reporter") and self.reporter is not None:
                    self._report(
                        x, potential, nbr_list, step, self._move_iteration, elapsed_step
                    )

                if self.save_traj_in_memory:
                    self.traj.append(x)

        log.debug("Finished running Langevin dynamics")

        # return the final state of the simulation as a sampler_state object
        import copy

        updated_sampler_state = copy.deepcopy(sampler_state)

        updated_sampler_state.positions = x
        updated_sampler_state.velocities = v
        updated_sampler_state.current_PRNG_key = key

        return updated_sampler_state, nbr_list

    def _wrap_and_rebuild_neighborlist(self, x: jnp.array, nbr_list: PairsBase):
        """
        Wrap the positions and rebuild the neighborlist if necessary.

        Parameters
        ----------
        x: jnp.array
            The positions of the particles.
        nbr_list: PairsBsse
            The neighborlist object.

        Returns
        -------
        x: jnp.array
            The wrapped positions.
        nbr_list: PairsBase
            The neighborlist object; this may or may not have been rebuilt.
        """

        x = nbr_list.space.wrap(x, self.box_vectors)
        # check if we need to rebuild the neighborlist after moving the particles
        if nbr_list.check(x):
            nbr_list.build(x, self.box_vectors)
        return x, nbr_list

    def _report(
        self,
        x: jnp.array,
        potential: NeuralNetworkPotential,
        nbr_list: PairsBase,
        step: int,
        iteration: int,
        elapsed_step: int,
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
                The current step in the move; this resets each iteration.
            iteration: int
                The number iterations the move has been called.
            elapsed_step: int,
                The total number of steps that have been taken in the simulation move.


        Returns:
            None
        """
        d = {
            "positions": x,
            "potential_energy": potential.compute_energy(x, nbr_list),
            "step": step,
            "iteration": iteration,
            "elapsed_step": elapsed_step,
        }
        if nbr_list is not None:
            d["box_vectors"] = nbr_list.box_vectors

        self.reporter.report(d)
