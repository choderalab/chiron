import copy
import time
from typing import List, Optional
from chiron.states import SamplerState, ThermodynamicState
import datetime
from loguru import logger as log
import numpy as np
from openmmtools.utils import time_it, with_timer
from chiron.neighbors import NeighborListNsqrd
from openmm import unit
from chiron.mcmc import MCMCMove
from openmmtools.multistate import MultiStateReporter


class MultiStateSampler(object):
    """
    Base class for samplers that sample multiple thermodynamic states using
    one or more replicas.

    This base class provides a general simulation facility for multistate from multiple
    thermodynamic states, allowing any set of thermodynamic states to be specified.
    If instantiated on its own, the thermodynamic state indices associated with each
    state are specified and replica mixing does not change any thermodynamic states,
    meaning that each replica remains in its original thermodynamic state.

    Parameters
    ----------
    mcmc_moves : MCMCMove or list of MCMCMove, optional
        The MCMCMove used to propagate the thermodynamic states. If a list of MCMCMoves,
        they will be assigned to the correspondent thermodynamic state on
        creation. If None is provided, Langevin dynamics with 2fm timestep, 5.0/ps collision rate,
        and 500 steps per iteration will be used.
    number_of_iterations : int or infinity, optional, default: 1
        The number of iterations to perform.

    locality : int > 0, optional, default None
        If None, the energies at all states will be computed for every replica each iteration.
        If int > 0, energies will only be computed for states ``range(max(0, state-locality), min(n_states, state+locality))``.

    Attributes
    ----------
    n_replicas
    n_states
    iteration
    mcmc_moves
    sampler_states
    metadata
    is_completed
    """

    def __init__(self, mcmc_moves=None, number_of_iterations=1, locality=None):
        # These will be set on initialization. See function
        # create() for explanation of single variables.
        self._thermodynamic_states = None
        self._unsampled_states = None
        self._sampler_states = None
        self._replica_thermodynamic_states = None
        self._iteration = None
        self._energy_thermodynamic_states = None
        self._neighborhoods = None
        self._energy_unsampled_states = None
        self._n_accepted_matrix = None
        self._n_proposed_matrix = None
        self._reporter = None
        self._metadata = None
        self._timing_data = dict()

        # Handling default propagator.
        if mcmc_moves is None:
            from .mcmc import LangevinDynamicsMove

            # This will be converted to a list in create().
            self._mcmc_moves = LangevinDynamicsMove(
                timestep=2.0 * unit.femtosecond,
                collision_rate=5.0 / unit.picosecond,
                n_steps=500,
            )
        else:
            self._mcmc_moves = copy.deepcopy(mcmc_moves)

        # Store constructor parameters. Everything is marked for internal
        # usage because any change to these attribute implies a change
        # in the storage file as well. Use properties for checks.
        self.number_of_iterations = number_of_iterations
        self._last_mbar_f_k = None
        self._last_err_free_energy = None

        # Store locality
        self.locality = locality

    @property
    def n_states(self):
        """The integer number of thermodynamic states (read-only)."""
        if self._thermodynamic_states is None:
            return 0
        else:
            return len(self._thermodynamic_states)

    @property
    def n_replicas(self):
        """The integer number of replicas (read-only)."""
        if self._sampler_states is None:
            return 0
        else:
            return len(self._sampler_states)

    @property
    def iteration(self):
        """The integer current iteration of the simulation (read-only).

        If the simulation has not been created yet, this is None.

        """
        return self._iteration

    @property
    def mcmc_moves(self):
        """A copy of the MCMCMoves list used to propagate the simulation.

        This can be set only before creation.

        """
        return copy.deepcopy(self._mcmc_moves)

    @property
    def sampler_states(self):
        """A copy of the sampler states list at the current iteration.

        This can be set only before running.
        """
        return copy.deepcopy(self._sampler_states)

    @property
    def is_periodic(self):
        """Return True if system is periodic, False if not, and None if not initialized"""
        if self._sampler_states is None:
            return None
        return self._thermodynamic_states[0].is_periodic

    @property
    def metadata(self):
        """A copy of the metadata dictionary passed on creation (read-only)."""
        return copy.deepcopy(self._metadata)

    @property
    def is_completed(self):
        """Check if we have reached any of the stop target criteria (read-only)"""
        return self._is_completed()

    def _compute_replica_energies(self, replica_id: int) -> np.ndarray:
        """
        Compute the energy for the replica in every ThermodynamicState.

        Parameters
        ----------
        replica_id : int
            The ID of the replica to compute energies for.

        Returns
        -------
        np.ndarray
            Array of energies for the specified replica across all thermodynamic states.
        """
        import jax.numpy as jnp
        from chiron.states import calculate_reduced_potential_at_states

        log.debug(f"{self._replica_thermodynamic_states=}")

        # Determine neighborhood
        state_index = self._replica_thermodynamic_states[replica_id]
        neighborhood = self._neighborhood(state_index)
        log.debug(f"{neighborhood=}")
        # Only compute energies of the sampled states over neighborhoods.
        energy_neighborhood_states = np.zeros(len(neighborhood))
        neighborhood_thermodynamic_states = [
            self._thermodynamic_states[n] for n in neighborhood
        ]

        # Retrieve sampler state associated to this replica.
        sampler_state = self._sampler_states[replica_id]
        log.debug(f"{sampler_state=}")
        # Compute energy for all thermodynamic states.
        from openmmtools.states import group_by_compatibility

        for energies, the_states in [
            (energy_neighborhood_states, neighborhood_thermodynamic_states),
        ]:
            # Group thermodynamic states by compatibility.
            compatible_groups, original_indices = group_by_compatibility(the_states)

            # Compute the reduced potentials of all the compatible states.
            for compatible_group, state_indices in zip(
                compatible_groups, original_indices
            ):
                # Compute and update the reduced potentials.
                compatible_energies = calculate_reduced_potential_at_states(
                    sampler_state, compatible_group, self.nbr_list
                )
                for energy_idx, state_idx in enumerate(state_indices):
                    energies[state_idx] = compatible_energies[energy_idx]

        # Return the new energies.
        log.info(f"Computed energies for replica {replica_id}")
        log.info(f"{energy_neighborhood_states=}")
        return energy_neighborhood_states

    def create(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        nbr_list: NeighborListNsqrd,
        reporter: MultiStateReporter,
        metadata: Optional[dict] = None,
    ):
        """Create new multistate sampler simulation.

        thermodynamic_states : List[ThermodynamicState]
            List of ThermodynamicStates to simulate, with one replica allocated per state.
        sampler_states : List[SamplerState]
            List of initial SamplerStates. The number of replicas is taken to be the number
            of sampler states provided.
        nbr_list : NeighborListNsqrd
            Neighbor list object to be used in the simulation.
        reporter : MultiStateReporter
            Reporter object to record simulation data.
        metadata : dict, optional
            Optional simulation metadata to be stored in the file.

        Raises
        ------
        RuntimeError
            If the lengths of thermodynamic_states and sampler_states are not equal.
        """
        # TODO: initialize reporter here
        # TODO: consider unsampled thermodynamic states for reweighting schemes

        # Ensure the number of thermodynamic states matches the number of sampler states
        if len(thermodynamic_states) != len(sampler_states):
            raise RuntimeError(
                "Number of thermodynamic states and sampler states must be equal."
            )

        self._allocate_variables(thermodynamic_states, sampler_states)
        self.nbr_list = nbr_list
        self._reporter = reporter
        self._reporter.open(mode="a")

    @classmethod
    def _default_initial_thermodynamic_states(
        cls,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
    ):
        """
        Create the initial_thermodynamic_states obeying the following rules:

        * ``len(thermodynamic_states) == len(sampler_states)``: 1-to-1 distribution
        """
        n_thermo = len(thermodynamic_states)
        n_sampler = len(sampler_states)
        assert n_thermo == n_sampler, "Must have 1-to-1 distribution of states"
        initial_thermo_states = np.arange(n_thermo, dtype=int)
        return initial_thermo_states

    def _allocate_variables(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        unsampled_thermodynamic_states: Optional[List[ThermodynamicState]] = None,
    ) -> None:
        """
        Allocate and initialize internal variables for the sampler.

        Parameters
        ----------
        thermodynamic_states : List[ThermodynamicState]
            A list of ThermodynamicState objects to be used in the sampler.
        sampler_states : List[SamplerState]
            A list of SamplerState objects for initializing the sampler.
        unsampled_thermodynamic_states : Optional[List[ThermodynamicState]], optional
            A list of additional ThermodynamicState objects that are not directly sampled but
            for which energies will be computed for reweighting schemes. Defaults to None,
            meaning no unsampled states are considered.

        Raises
        ------
        RuntimeError
            If the number of MCMC moves and ThermodynamicStates do not match.
        """

        # Save thermodynamic states. This sets n_replicas.
        self._thermodynamic_states = [
            copy.deepcopy(thermodynamic_state)
            for thermodynamic_state in thermodynamic_states
        ]

        # Deep copy sampler states.
        self._sampler_states = [
            copy.deepcopy(sampler_state) for sampler_state in sampler_states
        ]

        # Handle default unsampled thermodynamic states.
        self._unsampled_states = (
            copy.deepcopy(unsampled_thermodynamic_states)
            if unsampled_thermodynamic_states is not None
            else []
        )

        # Set initial thermodynamic state indices
        initial_thermodynamic_states = self._default_initial_thermodynamic_states(
            thermodynamic_states, sampler_states
        )
        self._replica_thermodynamic_states = np.array(
            initial_thermodynamic_states, np.int64
        )

        # Reset statistics.

        # _n_accepted_matrix[i][j] is the number of swaps proposed between thermodynamic states i and j.
        # _n_proposed_matrix[i][j] is the number of swaps proposed between thermodynamic states i and j.
        # Allocate memory for energy matrix. energy_thermodynamic_states[k][l]
        # is the reduced potential computed at the positions of SamplerState sampler_states[k]
        # and ThermodynamicState thermodynamic_states[l].

        self._n_accepted_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._n_proposed_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._energy_thermodynamic_states = np.zeros(
            [self.n_replicas, self.n_states], np.float64
        )
        self._neighborhoods = np.zeros([self.n_replicas, self.n_states], "i1")
        self._energy_unsampled_states = np.zeros(
            [self.n_replicas, len(self._unsampled_states)], np.float64
        )

        # Ensure there is an MCMCMove for each thermodynamic state.
        if isinstance(self._mcmc_moves, MCMCMove):
            self._mcmc_moves = [
                copy.deepcopy(self._mcmc_moves) for _ in range(self.n_states)
            ]
        elif len(self._mcmc_moves) != self.n_states:
            raise RuntimeError(
                f"The number of MCMCMoves ({len(self._mcmc_moves)}) and ThermodynamicStates ({self.n_states}) must be the same."
            )

        # Reset iteration counter.
        self._iteration = 0

    def _minimize_replica(
        self,
        replica_id: int,
        tolerance: unit.Quantity = 1.0 * unit.kilojoules_per_mole / unit.nanometers,
        max_iterations: int = 1_000,
    ) -> None:
        """
        Minimize the energy of a single replica.

        Parameters
        ----------
        replica_id : int
            The index of the replica to minimize.
        tolerance : unit.Quantity, optional
            The energy tolerance to which the system should be minimized.
            Defaults to 1.0 kilojoules/mole/nanometers.
        max_iterations : int, optional
            The maximum number of minimization iterations. Defaults to 1000.

        Notes
        -----
        The minimization modifies the SamplerState associated with the replica.
        """

        from chiron.minimze import minimize_energy

        # Retrieve thermodynamic and sampler states.
        thermodynamic_state = self._thermodynamic_states[
            self._replica_thermodynamic_states[replica_id]
        ]
        sampler_state = self._sampler_states[replica_id]

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.get_reduced_potential(sampler_state)
        log.debug(
            f"Replica {replica_id + 1}/{self.n_replicas}: initial energy {initial_energy:8.3f}kT"
        )

        # Perform minimization
        minimized_state = minimize_energy(
            sampler_state.x0,
            thermodynamic_state.potential.compute_energy,
            self.nbr_list,
            maxiter=max_iterations,
        )

        # Update the sampler state
        self._sampler_states[replica_id].x0 = minimized_state.params

        # Compute and log final energy
        final_energy = thermodynamic_state.get_reduced_potential(sampler_state)
        log.debug(
            f"Replica {replica_id + 1}/{self.n_replicas}: final energy {final_energy:8.3f}kT"
        )

    def minimize(
        self,
        tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometers,
        max_iterations: int = 1_000,
    ) -> None:
        """
        Minimize all replicas in the sampler.

        This method minimizes the positions of all replicas to the nearest local
        minimum of the potential energy surface. The minimized positions are stored
        at the end of the process.

        Parameters
        ----------
        tolerance : unit.Quantity, optional
            The energy tolerance for the minimization. Default is 1.0 kJ/mol/nm.
        max_iterations : int, optional
            The maximum number of iterations for the minimization process.
            Default is 1000.

        Raises
        ------
        RuntimeError
            If the simulation has not been created before calling this method.
        """

        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError(
                "Cannot minimize replicas. The simulation must be created first."
            )

        log.debug("Minimizing all replicas...")

        # Iterate over all replicas and minimize them
        for replica_id in range(self.n_replicas):
            self._minimize_replica(replica_id, tolerance, max_iterations)

    def _equilibration_timings(self, timer, iteration: int, n_iterations: int):
        iteration_time = timer.stop("Equilibration Iteration")
        partial_total_time = timer.partial("Run Equilibration")
        time_per_iteration = partial_total_time / iteration
        estimated_time_remaining = time_per_iteration * (n_iterations - iteration)
        estimated_total_time = time_per_iteration * n_iterations
        estimated_finish_time = time.time() + estimated_time_remaining
        # TODO: Transmit timing information

        log.info(f"Iteration took {iteration_time:.3f}s.")
        if estimated_time_remaining != float("inf"):
            log.info(
                "Estimated completion (of equilibration only) in {}, at {} (consuming total wall clock time {}).".format(
                    str(datetime.timedelta(seconds=estimated_time_remaining)),
                    time.ctime(estimated_finish_time),
                    str(datetime.timedelta(seconds=estimated_total_time)),
                )
            )

    def equilibrate(
        self, n_iterations: int, mcmc_moves: Optional[List[MCMCMove]] = None
    ):
        """
        Equilibrate all replicas in the sampler.

        This method equilibrates the system by running a specified number of
        MCMC iterations. The equilibration uses either the provided MCMC moves
        or the default ones set during initialization.

        Parameters
        ----------
        n_iterations : int
            The number of equilibration iterations to perform.
        mcmc_moves : Optional[List[mcmc.MCMCMove]], optional
            A list of MCMCMove objects to use for equilibration. If None, the
            MCMC moves used in production will be used. Defaults to None.

        Raises
        ------
        RuntimeError
            If the simulation has not been created before calling this method.
        """
        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError(
                "Cannot equilibrate replicas. The simulation must be created first."
            )

        # Use production MCMC moves if none are provided
        mcmc_moves = mcmc_moves or self._mcmc_moves

        # Make sure there is one MCMCMove per thermodynamic state.
        if isinstance(mcmc_moves, MCMCMove):
            mcmc_moves = [copy.deepcopy(mcmc_moves) for _ in range(self.n_states)]

        if len(mcmc_moves) != self.n_states:
            raise RuntimeError(
                f"The number of MCMCMoves ({len(self._mcmc_moves)}) and ThermodynamicStates ({self.n_states}) for equilibration must be the same."
            )
        from openmmtools.utils import Timer

        timer = Timer()
        timer.start("Run Equilibration")

        # Temporarily set the equilibration MCMCMoves.
        production_mcmc_moves = self._mcmc_moves
        self._mcmc_moves = mcmc_moves

        for iteration in range(1, n_iterations + 1):
            log.info(f"Equilibration iteration {iteration}/{n_iterations}")
            timer.start("Equilibration Iteration")

            # NOTE: Unlike run(), do NOT increment iteration counter.
            # self._iteration += 1

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Update thermodynamic states
            self._replica_thermodynamic_states = self._mix_replicas()

            # Computing timing information
            self._equilibration_timings(
                timer, iteration=iteration, n_iterations=n_iterations
            )
        timer.report_timing()

        # Restore production MCMCMoves.
        self._mcmc_moves = production_mcmc_moves

        # TODO: Update stored positions.

    def _propagate_replica(self, replica_id: int):
        """
        Propagate the state of a single replica.

        This method applies the MCMC move to the replica to change its state
        according to the specified thermodynamic state.

        Parameters
        ----------
        replica_id : int
            The index of the replica to propagate.
        Raises
        ------
        RuntimeError
            If an error occurs during the propagation of the replica.
        """
        # Retrieve thermodynamic, sampler states, and MCMC move of this replica.
        # Retrieve thermodynamic and sampler states for the replica
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        sampler_state = self._sampler_states[replica_id]

        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        mcmc_move = self._mcmc_moves[thermodynamic_state_id]

        # Apply MCMC move.
        try:
            mcmc_move.run(sampler_state, thermodynamic_state)
        except Exception as e:
            log.warning(e)
            raise e

    def _perform_swap_proposals(self):
        """
        Perform swap proposals between replicas.

        Placeholder method for replica swapping logic. Subclasses should
        override this method with specific swapping algorithms.

        Returns
        -------
        np.ndarray
            An array of updated thermodynamic state indices for each replica.
        """

        # Placeholder implementation, should be overridden by subclasses
        # For this example, we'll just return the current state indices
        return self._replica_thermodynamic_states

    def _mix_replicas(self) -> np.ndarray:
        """
        Propose and execute swaps between replicas.

        This method is responsible for enhancing sampling efficiency by proposing
        swaps between different thermodynamic states of the replicas. The actual
        swapping algorithm depends on the specific subclass implementation.

        Returns
        -------
        np.ndarray
            An array of updated thermodynamic state indices for each replica.
        """

        log.debug("Mixing replicas (does nothing for MultiStateSampler)...")

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform replica mixing (swap proposals and acceptances)
        # The actual swapping logic would depend on subclass implementations
        # Here, we assume a placeholder implementation
        new_replica_states = self._perform_swap_proposals()

        # Calculate swap acceptance statistics
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            swap_fraction_accepted = n_swaps_accepted / n_swaps_proposed
        log.debug(
            f"Accepted {n_swaps_accepted}/{n_swaps_proposed} attempted swaps ({swap_fraction_accepted * 100.0:.1f}%)"
        )
        return new_replica_states

    @with_timer("Propagating all replicas")
    def _propagate_replicas(self):
        """Propagate all replicas."""

        log.debug("Propagating all replicas...")

        for i in range(self.n_replicas):
            self._propagate_replica(i)

    def _neighborhood(self, state_index):
        """Compute the states in the local neighborhood determined by self.locality

        Parameters
        ----------
        state_index : int
            The current state

        Returns
        -------
        neighborhood : list of int
            The states in the local neighborhood
        """
        if self.locality is None:
            # Global neighborhood
            return list(range(0, self.n_states))
        else:
            # Local neighborhood specified by 'locality'
            return list(
                range(
                    max(0, state_index - self.locality),
                    min(self.n_states, state_index + self.locality + 1),
                )
            )

    @with_timer("Computing energy matrix")
    def _compute_energies(self):
        """Compute energies of all replicas at all states."""

        # Determine neighborhoods (all nodes)
        self._neighborhoods[:, :] = False
        for replica_index, state_index in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(state_index)
            self._neighborhoods[replica_index, neighborhood] = True

        # Calculate energies for all replicas.
        new_energies, replica_ids = [], []
        for replica_id in range(self.n_replicas):
            new_energy = self._compute_replica_energies(replica_id)
            new_energies.append(new_energy)
            replica_ids.append(replica_id)

        # Update energy matrices.
        for replica_id, energies in zip(replica_ids, new_energies):
            energy_thermodynamic_states = energies  # Unpack.
            neighborhood = self._neighborhood(
                self._replica_thermodynamic_states[replica_id]
            )
            self._energy_thermodynamic_states[
                replica_id, neighborhood
            ] = energy_thermodynamic_states

    def _is_completed(self, iteration_limit=None):
        """Check if we have reached any of the stop target criteria.

        Parameters
        ----------
        iteration_limit : int, optional
            If specified, the simulation will stop if the iteration counter reaches this value.

        Returns
        -------
        is_completed : bool
            If True, the simulation is completed and should be terminated.
        """
        if iteration_limit is not None and self._iteration >= iteration_limit:
            log.info(
                f"Reached iteration limit {iteration_limit} (current iteration {self._iteration})"
            )
            return True
        return False

    def run(self, n_iterations=None):
        """Run the replica-exchange simulation.

        This runs at most ``number_of_iterations`` iterations.

        Parameters
        ----------
        n_iterations : int, optional
           If specified, only at most the specified number of iterations
           will be run (default is None).
        """
        # If this is the first iteration, compute and store the
        # starting energies of the minimized/equilibrated structures.

        log.info("Running simulation...")
        if self._iteration == 0:
            try:
                self._compute_energies()
            except Exception as e:
                log.critical(e)
                raise e

            self._reporter.write_energies(
                energy_thermodynamic_states=self._energy_thermodynamic_states,
                energy_neighborhoods=self._neighborhoods,
                energy_unsampled_states=self._energy_unsampled_states,
                iteration=self._iteration,
            )

        from openmmtools.utils import Timer

        timer = Timer()
        timer.start("Run ReplicaExchange")
        run_initial_iteration = self._iteration

        # Handle default argument and determine number of iterations to run.
        if n_iterations is None:
            iteration_limit = self.number_of_iterations
        else:
            iteration_limit = min(
                self._iteration + n_iterations, self.number_of_iterations
            )

        # Main loop.
        while not self._is_completed(iteration_limit):
            # Increment iteration counter.
            self._iteration += 1

            log.info("-" * 80)
            log.info(f"Iteration {self._iteration}/{iteration_limit}")
            log.info("-" * 80)
            timer.start("Iteration")

            # Update thermodynamic states
            self._replica_thermodynamic_states = self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Write iteration to storage file
            self._report_iteration()

            # TODO: Update analysis
            # self._update_analysis()

            # Computing and transmitting timing information
            iteration_time = timer.stop("Iteration")
            partial_total_time = timer.partial("Run ReplicaExchange")
            self._update_timing(
                iteration_time,
                partial_total_time,
                run_initial_iteration,
                iteration_limit,
            )

            # Log timing data as info level -- useful for users by default
            log.info(
                "Iteration took {:.3f}s.".format(self._timing_data["iteration_seconds"])
            )
            if self._timing_data["estimated_time_remaining"] != float("inf"):
                log.info(
                    f"Estimated completion in {self._timing_data['estimated_time_remaining']}, at {self._timing_data['estimated_localtime_finish_date']} (consuming total wall clock time {self._timing_data['estimated_total_time']})."
                )

            # Perform sanity checks to see if we should terminate here.
            self._check_nan_energy()

    @with_timer("Writing iteration information to storage")
    def _report_iteration(self):
        """Store positions, states, and energies of current iteration.n"""
        # Call report_iteration_items for a subclass-friendly function
        self._report_iteration_items()
        self._reporter.write_timestamp(self._iteration)
        self._reporter.write_last_iteration(self._iteration)

    def _report_iteration_items(self):
        """
        Sub-function of :func:`_report_iteration` which handles all the actual individual item reporting in a
        sub-class friendly way. The final actions of writing timestamp, last-good-iteration, and syncing
        should be left to the :func:`_report_iteration` and subclasses should extend this function instead
        """
        self._reporter.write_sampler_states(self._sampler_states, self._iteration)
        self._reporter.write_replica_thermodynamic_states(
            self._replica_thermodynamic_states, self._iteration
        )
        self._reporter.write_mcmc_moves(
            self._mcmc_moves
        )  # MCMCMoves can store internal statistics.
        self._reporter.write_energies(
            self._energy_thermodynamic_states,
            self._neighborhoods,
            self._energy_unsampled_states,
            self._iteration,
        )
        self._reporter.write_mixing_statistics(
            self._n_accepted_matrix, self._n_proposed_matrix, self._iteration
        )

    def _update_timing(
        self, iteration_time, partial_total_time, run_initial_iteration, iteration_limit
    ):
        """
        Function that computes and transmits timing information to reporter.

        Parameters
        ----------
        iteration_time : float
            Time took in the iteration.
        partial_total_time : float
            Partial total time elapsed.
        run_initial_iteration : int
            Iteration where to start/resume the simulation.
        iteration_limit : int
            Hard limit on number of iterations to be run by the sampler.
        """
        self._timing_data["iteration_seconds"] = iteration_time
        self._timing_data["average_seconds_per_iteration"] = partial_total_time / (
            self._iteration - run_initial_iteration
        )
        estimated_timedelta_remaining = datetime.timedelta(
            seconds=self._timing_data["average_seconds_per_iteration"]
            * (iteration_limit - self._iteration)
        )
        estimated_finish_date = datetime.datetime.now() + estimated_timedelta_remaining
        self._timing_data["estimated_time_remaining"] = str(
            estimated_timedelta_remaining
        )  # Putting it in dict as str
        self._timing_data[
            "estimated_localtime_finish_date"
        ] = estimated_finish_date.strftime("%Y-%b-%d-%H:%M:%S")
        total_time_in_seconds = datetime.timedelta(
            seconds=self._timing_data["average_seconds_per_iteration"] * iteration_limit
        )
        self._timing_data["estimated_total_time"] = str(total_time_in_seconds)

        # Estimate performance
        moves_iterator = self._flatten_moves_iterator()
        # Only consider "dynamic" moves (timestep and n_steps attributes)
        moves_times = [
            move.timestep.value_in_unit(unit.nanosecond) * move.n_steps
            for move in moves_iterator
            if hasattr(move, "timestep") and hasattr(move, "n_steps")
        ]
        iteration_simulated_nanoseconds = sum(moves_times)
        seconds_in_a_day = (1 * unit.day).value_in_unit(unit.seconds)
        self._timing_data["ns_per_day"] = iteration_simulated_nanoseconds / (
            self._timing_data["average_seconds_per_iteration"] / seconds_in_a_day
        )

    def _flatten_moves_iterator(self):
        """Recursively flatten MCMC moves. Handles the cases where each move can be a set of moves, for example with
        SequenceMove or WeightedMove objects."""

        def flatten(iterator):
            try:
                yield from [
                    inner_move for move in iterator for inner_move in flatten(move)
                ]
            except TypeError:  # Inner object is not iterable, finish flattening.
                yield iterator

        return flatten(self.mcmc_moves)

    def _check_nan_energy(self):
        """Checks that energies are finite and abort otherwise.

        Checks both sampled and unsampled thermodynamic states.

        """
        # Find faulty replicas to create error message.
        nan_replicas = []

        # Check sampled thermodynamic states first.
        state_type = "thermodynamic state"
        for replica_id, state_id in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(state_id)
            energies_neighborhood = self._energy_thermodynamic_states[
                replica_id, neighborhood
            ]
            if np.any(np.isnan(energies_neighborhood)):
                nan_replicas.append((replica_id, energies_neighborhood))

        # If there are no NaNs in energies, look for NaNs in the unsampled states energies.
        if (len(nan_replicas) == 0) and (self._energy_unsampled_states.shape[1] > 0):
            state_type = "unsampled thermodynamic state"
            for replica_id in range(self.n_replicas):
                if np.any(np.isnan(self._energy_unsampled_states[replica_id])):
                    nan_replicas.append(
                        (replica_id, self._energy_unsampled_states[replica_id])
                    )

        # Raise exception if we have found some NaN energies.
        if len(nan_replicas) > 0:
            # Log failed replica, its thermo state, and the energy matrix row.
            err_msg = "NaN encountered in {} energies for the following replicas and states".format(
                state_type
            )
            for replica_id, energy_row in nan_replicas:
                err_msg += "\n\tEnergies for positions at replica {} (current state {}): {} kT".format(
                    replica_id,
                    self._replica_thermodynamic_states[replica_id],
                    energy_row,
                )
            log.critical(err_msg)
            raise RuntimeError(err_msg)
