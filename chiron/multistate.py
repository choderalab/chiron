import copy
from typing import List, Optional
from chiron.states import SamplerState, ThermodynamicState
import datetime
from loguru import logger as log
import numpy as np
from openmmtools.utils import with_timer
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

    locality : int > 0, optional, default None
        If None, the energies at all states will be computed for every replica each iteration.
        If int > 0, energies will only be computed for states ``range(max(0, state-locality), min(n_states, state+locality))``.

    Attributes
    ----------
    n_replicas
    n_states
    mcmc_moves
    sampler_states
    metadata
    is_completed
    """

    def __init__(
        self,
        mcmc_moves=None,
        locality=None,
        online_analysis_interval=5,
    ):
        # These will be set on initialization. See function
        # create() for explanation of single variables.
        self._thermodynamic_states = None
        self._unsampled_states = None
        self._sampler_states = None
        self._replica_thermodynamic_states = None
        self._iteration = None
        self._energy_thermodynamic_states = None
        self._energy_thermodynamic_states_for_each_iteration = None
        self._neighborhoods = None
        self._energy_unsampled_states = None
        self._n_accepted_matrix = None
        self._n_proposed_matrix = None
        self._reporter = None
        self._metadata = None
        self._online_analysis_interval = online_analysis_interval
        self._timing_data = dict()
        self.free_energy_estimator = None
        self._traj = None

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

        # Only compute energies of the sampled states over neighborhoods.
        thermodynamic_states = [
            self._thermodynamic_states[n] for n in range(self.n_states)
        ]
        # Retrieve sampler state associated to this replica.
        sampler_state = self._sampler_states[replica_id]
        # Compute energy for all thermodynamic states.
        return calculate_reduced_potential_at_states(
            sampler_state, thermodynamic_states, self.nbr_list
        )

    def create(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        nbr_list: NeighborListNsqrd,
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
        metadata : dict, optional
            Optional simulation metadata to be stored in the file.

        Raises
        ------
        RuntimeError
            If the lengths of thermodynamic_states and sampler_states are not equal.
        """
        # TODO: initialize reporter here
        # TODO: consider unsampled thermodynamic states for reweighting schemes
        self.free_energy_estimator = "mbar"

        # Ensure the number of thermodynamic states matches the number of sampler states
        if len(thermodynamic_states) != len(sampler_states):
            raise RuntimeError(
                "Number of thermodynamic states and sampler states must be equal."
            )

        self._allocate_variables(thermodynamic_states, sampler_states)
        self.nbr_list = nbr_list
        self._reporter = None

    def _allocate_variables(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
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

        assert len(self._thermodynamic_states) == len(self._sampler_states)
        # Set initial thermodynamic state indices
        initial_thermodynamic_states = np.arange(
            len(self._thermodynamic_states), dtype=int
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
        self._traj = [[] for _ in range(self.n_replicas)]
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
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        sampler_state = self._sampler_states[replica_id]

        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        mcmc_move = self._mcmc_moves[thermodynamic_state_id]
        # Apply MCMC move.
        mcmc_move.run(sampler_state, thermodynamic_state)
        self._traj[replica_id].append(sampler_state.x0)

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

    @with_timer("Propagating all replicas")
    def _propagate_replicas(self) -> None:
        """
        Propagate all replicas through their respective MCMC moves.

        This method iterates over all replicas and applies the corresponding MCMC move
        to each one, based on its current thermodynamic state.
        """

        log.debug("Propagating all replicas...")

        for replica_id in range(self.n_replicas):
            self._propagate_replica(replica_id)

    @with_timer("Computing energy matrix")
    def _compute_energies(self) -> None:
        """
        Compute the energies of all replicas at all thermodynamic states.

        This method calculates the energy for each replica in every thermodynamic state,
        considering the defined neighborhoods to optimize the computation. The energies
        are stored in the internal energy matrix of the sampler.
        """

        log.debug("Computing energy matrix for all replicas...")
        # Initialize the energy matrix and neighborhoods
        self._energy_thermodynamic_states = np.zeros((self.n_replicas, self.n_states))

        # Calculate energies for each replica
        for replica_id in range(self.n_replicas):
            # Compute and store energies for the neighborhood states
            self._energy_thermodynamic_states[
                replica_id, :
            ] = self._compute_replica_energies(replica_id)

    def _is_completed(self, iteration_limit: Optional[int] = None) -> bool:
        """
        Determine if the sampling process has met its completion criteria.

        This method checks if the simulation has reached a specified iteration limit
        or any other predefined stopping condition.

        Parameters
        ----------
        iteration_limit : Optional[int], default=None
            An optional iteration limit. If specified, the method checks if the
            current iteration number has reached this limit.

        Returns
        -------
        bool
            True if the simulation has completed based on the stopping criteria,
            False otherwise.
        """

        # Check if iteration limit has been reached
        if iteration_limit is not None and self._iteration >= iteration_limit:
            log.info(
                f"Reached iteration limit {iteration_limit} (current iteration {self._iteration})"
            )
            return True

        # Additional stopping criteria can be implemented here

        return False

    def _update_run_progress(self, timer, run_initial_iteration, iteration_limit):
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

    def run(self, n_iterations: int = 10) -> None:
        """
        Execute the replica-exchange simulation.

        Run the simulation for a specified number of iterations. If no number is
        specified, it runs for the number of iterations set during the initialization
        of the sampler.

        Parameters
        ----------
        n_iterations : int, default=10
            The number of iterations to run.

        Raises
        ------
        RuntimeError
            If an error occurs during the computation of energies.
        """

        # If this is the first iteration, compute and store the
        # starting energies of the minimized/equilibrated structures.
        self.number_of_iterations = n_iterations

        log.info("Running simulation...")
        self._energy_thermodynamic_states_for_each_iteration_in_run = np.zeros(
            [self.n_replicas, self.n_states, n_iterations + 1], np.float64
        )

        # Initialize energies if this is the first iteration
        if self._iteration == 0:
            self._compute_energies()
            # store energies for mbar analysis
            self._energy_thermodynamic_states_for_each_iteration_in_run[
                :, :, self._iteration
            ] = self._energy_thermodynamic_states
            # TODO report energies

        from openmmtools.utils import Timer

        timer = Timer()
        timer.start("Run ReplicaExchange")

        iteration_limit = n_iterations

        # start the sampling loop
        log.debug(f"{iteration_limit=}")
        while not self._is_completed(iteration_limit):
            # Increment iteration counter.
            self._iteration += 1

            log.info("-" * 80)
            log.info(f"Iteration {self._iteration}/{iteration_limit}")
            log.info("-" * 80)
            timer.start("Iteration")

            # Update thermodynamic states
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Add energies to the energy matrix
            self._energy_thermodynamic_states_for_each_iteration_in_run[
                :, :, self._iteration
            ] = self._energy_thermodynamic_states
            # Write iteration to storage file
            # TODO
            # self._report_iteration()

            # Update analysis
            self._update_analysis()

    def _report_iteration(self):
        """Store positions, states, and energies of current iteration."""

        # TODO: write energies

        # TODO: write trajectory

        # TODO: write mixing statistics
        self._reporter.write_energies(
            self._energy_thermodynamic_states,
            self._neighborhoods,
            self._energy_unsampled_states,
            self._iteration,
        )

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

    def _update_analysis(self):
        """Update analysis of free energies"""

        if self._online_analysis_interval is None:
            log.debug("No online analysis requested")
            # Perform no analysis and exit function
            return

        # Perform offline free energy estimate if requested
        if self.free_energy_estimator == "mbar":
            self._last_err_free_energy = self._mbar_analysis()

        return

    def _mbar_analysis(self):
        """
        Perform mbar analysis
        """
        from pymbar import MBAR

        self._last_mbar_f_k_offline = np.zeros(len(self._thermodynamic_states))

        log.debug(
            f"{self._energy_thermodynamic_states_for_each_iteration_in_run.shape=}"
        )
        log.debug(f"{self.n_states=}")
        u_kn = self._energy_thermodynamic_states_for_each_iteration_in_run
        log.debug(f"{self._iteration=}")
        N_k = [self._iteration] * self.n_states
        log.debug(f"{N_k=}")
        mbar = MBAR(u_kn=u_kn, N_k=N_k)
        log.debug(mbar.f_k)
        self._last_mbar_f_k_offline = mbar.f_k
