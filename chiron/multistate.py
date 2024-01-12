from typing import List, Optional, Union
from chiron.states import SamplerState, ThermodynamicState
from chiron.neighbors import NeighborListNsqrd
from openmm import unit
import numpy as np
from chiron.mcmc import MCMCMove
from chiron.reporters import MultistateReporter


class MultiStateSampler:
    """
    Base class for samplers that sample multiple thermodynamic states using
    one or more replicas.

    This base class provides a general simulation facility for multistate from multiple
    thermodynamic states, allowing any set of thermodynamic states to be specified.
    If instantiated on its own, the thermodynamic state indices associated with each
    state are specified and replica mixing does not change any thermodynamic states,
    meaning that each replica remains in its original thermodynamic state.
    """

    def __init__(
        self, mcmc_moves: Union[MCMCMove, List[MCMCMove]], reporter: MultistateReporter
    ):
        """
        Parameters
        ----------
        mcmc_moves : MCMCMove or list of MCMCMove
            The MCMCMove used to propagate the thermodynamic states. If a list of MCMCMoves,
            they will be assigned to the correspondent thermodynamic state on
            creation.
        reporter : MultistateReporter
        Attributes
        ----------
        n_replicas
        n_states
        mcmc_moves
        sampler_states
        is_completed
        """
        import copy

        # These will be set on initialization. See function
        # create() for explanation of single variables.
        self._thermodynamic_states = None
        self._unsampled_states = None
        self._sampler_states = None
        self._replica_thermodynamic_states = None
        self._iteration = None
        self._energy_thermodynamic_states = None
        self._neighborhoods = None
        self._n_accepted_matrix = None
        self._n_proposed_matrix = None
        self._reporter = reporter
        self._metadata = None
        self._timing_data = dict()

        self._mcmc_moves = copy.deepcopy(mcmc_moves)

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
        import copy

        return copy.deepcopy(self._mcmc_moves)

    @property
    def sampler_states(self):
        """A copy of the sampler states list at the current iteration.

        This can be set only before running.
        """
        import copy

        return copy.deepcopy(self._sampler_states)

    @property
    def is_periodic(self):
        """Return True if system is periodic, False if not, and None if not initialized"""
        if self._sampler_states is None:
            return None
        return self._thermodynamic_states[0].is_periodic

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
    ):
        """Create new multistate sampler simulation.

        thermodynamic_states : List[ThermodynamicState]
            List of ThermodynamicStates to simulate, with one replica allocated per state.
        sampler_states : List[SamplerState]
            List of initial SamplerStates. The number of replicas is taken to be the number
            of sampler states provided.
        nbr_list : NeighborListNsqrd
            Neighbor list object to be used in the simulation.

        Raises
        ------
        RuntimeError
            If the lengths of thermodynamic_states and sampler_states are not equal.
        """
        # TODO: initialize reporter here
        # TODO: consider unsampled thermodynamic states for reweighting schemes
        self._online_estimator = None

        from chiron.analysis import MBAREstimator
        from chiron.reporters import MultistateReporter

        n_thermodynamic_states = len(thermodynamic_states)
        n_sampler_states = len(sampler_states)

        self._offline_estimator = MBAREstimator()

        # Ensure the number of thermodynamic states matches the number of sampler states
        if n_thermodynamic_states != n_sampler_states:
            raise RuntimeError(
                "Number of thermodynamic states and sampler states must be equal."
            )

        self._allocate_variables(thermodynamic_states, sampler_states)
        self.nbr_list = nbr_list
        self._reporter = MultistateReporter()

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
        import copy
        import numpy as np

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
        # energy_thermodynamic_states[k][l] is the reduced potential computed at the positions of
        # SamplerState sampler_states[k] and ThermodynamicState thermodynamic_states[l].

        self._n_accepted_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._n_proposed_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._energy_thermodynamic_states = np.zeros(
            [self.n_replicas, self.n_states], np.float64
        )
        self._traj = [[] for _ in range(self.n_replicas)]
        # Ensure there is an MCMCMove for each thermodynamic state.
        from chiron.mcmc import MCMCMove

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
        from loguru import logger as log

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
        from loguru import logger as log

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
        from loguru import logger as log

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

    def _propagate_replicas(self) -> None:
        """
        Propagate all replicas through their respective MCMC moves.

        This method iterates over all replicas and applies the corresponding MCMC move
        to each one, based on its current thermodynamic state.
        """
        from loguru import logger as log

        log.debug("Propagating all replicas...")

        for replica_id in range(self.n_replicas):
            self._propagate_replica(replica_id)

    def _compute_energies(self) -> None:
        """
        Compute the energies of all replicas at all thermodynamic states.

        This method calculates the energy for each replica in every thermodynamic state,
        considering the defined neighborhoods to optimize the computation. The energies
        are stored in the internal energy matrix of the sampler.
        """
        from loguru import logger as log

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
        from loguru import logger as log

        # Check if iteration limit has been reached
        if iteration_limit is not None and self._iteration >= iteration_limit:
            log.info(
                f"Reached iteration limit {iteration_limit} (current iteration {self._iteration})"
            )
            return True

        # Additional stopping criteria can be implemented here

        return False

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
        from loguru import logger as log

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
            self._report_iteration()

        # start the sampling loop
        log.debug(f"{n_iterations=}")
        while not self._is_completed(n_iterations):
            # Increment iteration counter.
            self._iteration += 1

            log.info("-" * 80)
            log.info(f"Iteration {self._iteration}/{n_iterations}")
            log.info("-" * 80)

            # Update thermodynamic states
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Write iteration to storage file
            self._report_iteration()

            # Update analysis
            self._update_analysis()

    def _report_energy_matrix(self):
        from loguru import logger as log

        log.debug("Reporting energy per thermodynamic state...")
        self._reporter.report({"u_kn": self._energy_thermodynamic_states.T})

    def _report_positions(self):
        """Store positions of current iteration."""
        from loguru import logger as log

        log.debug("Reporting positions...")
        for replica_id in range(self.n_replicas):
            self._reporter.report(
                {
                    "positions": {
                        "xyz": self._sampler_states[replica_id].x0,
                        "replica_id": replica_id,
                    }
                }
            )

    def _report(self, property: str):
        """
        Report a property of the simulation.

        Parameters
        ----------
        property : str
            The property to report. Options are 'positions', 'states', 'energies',
            'trajectory', 'mixing_statistics', and 'all'.
        """
        from loguru import logger as log

        log.debug(f"Reporting {property}...")
        match property:
            case "positions":
                self._report_positions()
            case "states":
                pass
            case "u_kn":
                self._report_energy_matrix()
            case "trajectory":
                pass
            case "mixing_statistics":
                pass
                # reporter.write_mixing_statistics()
        a = 7

    def _report_iteration(self):
        """Store positions, states, and energies of current iteration."""

        for property in self._reporter.properties_to_report:
            self._report(property)

    def _update_analysis(self):
        """Update analysis of free energies"""
        from loguru import logger as log

        # Perform offline free energy estimate if requested
        if self._offline_estimator:
            log.debug("Performing offline free energy estimate...")
            N_k = [self._iteration] * self.n_states
            u_kn = self._reporter.get_property("u_kn")
            u_kn = np.transpose(
                u_kn, (2, 1, 0)
            )  # shape: n_states, n_replicas, n_iterations
            self._offline_estimator.initialize(
                u_kn=u_kn,
                N_k=N_k,
            )
            log.debug(self._offline_estimator.f_k)
        elif self._online_estimator:
            log.debug("Performing online free energy estimate...")
            self._online_estimator.update()
        else:
            raise RuntimeError("No free energy estimator provided.")

    @property
    def f_k(self):
        if self._offline_estimator:
            return self._offline_estimator.f_k
        elif self._online_estimator:
            return self._online_estimator.f_k
        else:
            raise RuntimeError("No free energy estimator found.")
