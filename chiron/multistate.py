from typing import List, Optional, Union
from chiron.states import SamplerState, ThermodynamicState
from chiron.neighbors import PairsBase
from openmm import unit
import numpy as np
from chiron.mcmc import MCMCMove, MCMCSampler
from chiron.reporters import MultistateReporter


class MultiStateSampler:
    """
    A sampler for simulating multiple thermodynamic states using replicas.

    This class provides a general simulation facility for sampling from multiple
    thermodynamic states. It allows specifying any set of thermodynamic states.
    If instantiated on its own, the thermodynamic state indices associated with
    each state are specified, and replica mixing does not change any thermodynamic states,
    meaning that each replica remains in its original thermodynamic state.

    Attributes
    ----------
    n_states : int
        Number of thermodynamic states (read-only).
    n_replicas : int
        Number of replicas (read-only).
    iteration : int
        Current iteration of the simulation (read-only).
    mcmc_sampler : MCMCSampler
        MCMC sampler used to propagate the simulation.
    sampler_states : List[SamplerState]
        Sampler states list at the current iteration.
    is_periodic : bool
        True if system is periodic, False if not, None if not initialized.
    is_completed : bool
        Check if the sampler has reached any stop target criteria (read-only).

    Methods
    -------
    create(thermodynamic_states: List[ThermodynamicState], sampler_states: List[SamplerState], nbr_lists: List[PairsBase])
        Creates a new multistate sampler simulation.
    minimize(tolerance: unit.Quantity = 1.0 * unit.kilojoules_per_mole / unit.nanometers, max_iterations: int = 1000)
        Minimizes all replicas in the sampler.
    run(n_iterations: int = 10)
        Executes the replica-exchange simulation for a specified number of iterations.

    """

    def __init__(
        self,
        mcmc_sampler: MCMCSampler,
        reporter: MultistateReporter,
    ):
        """
        Initialize the MultiStateSampler.

        Parameters
        ----------
        mcmc_sampler : MCMCSampler
            The MCMCSampler used to propagate the thermodynamic states.
        reporter : MultistateReporter
            The reporter used to store the simulation data.
        """

        import copy
        from chiron.analysis import MBAREstimator

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
        self._nbr_lists = None

        self._reporter = reporter  # NOTE: reporter needs to be putlic, API change ahead
        self._metadata = None
        self._mcmc_sampler = copy.deepcopy(mcmc_sampler)
        self._online_estimator = None
        self._offline_estimator = MBAREstimator()

    @property
    def n_states(self) -> int:
        """
        Get the number of thermodynamic states in the sampler.

        Returns
        -------
        int
            The number of thermodynamic states.
        """
        if self._thermodynamic_states is None:
            return 0
        else:
            return len(self._thermodynamic_states)

    @property
    def n_replicas(self) -> int:
        """
        Get the number of replicas in the sampler.

        Returns
        -------
        int
            The number of replicas.
        """
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
    def mcmc_sampler(self):
        """A copy of the MCMCSampler used to propagate the simulation.

        This can be set only before creation.

        """
        import copy

        return copy.deepcopy(self._mcmc_sampler)

    @property
    def sampler_states(self) -> Optional[List[SamplerState]]:
        """
        Get a copy of the sampler states list at the current iteration.

        This property can only be set before running the simulation.

        Returns
        -------
        Optional[List[SamplerState]]
            The list of sampler states at the current iteration, or None if not set.
        """
        if self._sampler_states is None:
            return None
        import copy

        return copy.deepcopy(self._sampler_states)

    @property
    def is_periodic(self):
        """
        Determine if the system is periodic.

        Returns
        -------
        Optional[bool]
            True if the system is periodic, False if not, and None if not initialized.
        """
        if self._sampler_states is None:
            return None
        return self.is_periodic

    @property
    def is_completed(self):
        """Check if we have reached any of the stop target criteria (read-only)"""
        return self._is_completed()

    def _compute_replica_energies(self, replica_id: int) -> np.ndarray:
        """
        Compute the energy of a replica across all thermodynamic states.

        Parameters
        ----------
        replica_id : int
            The index of the replica for which to compute energies.

        Returns
        -------
        np.ndarray
            An array of energies for the replica across all thermodynamic states.
        """
        from chiron.states import calculate_reduced_potential_at_states

        # Retrieve sampler state associated to this replica.
        sampler_state = self._sampler_states[replica_id]
        nbr_list = self._sampler_states[replica_id]
        # Compute energy for all thermodynamic states.
        energies = calculate_reduced_potential_at_states(
            sampler_state, self._thermodynamic_states, nbr_list
        )
        return energies

    def create(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        nbr_lists: List[PairsBase],
    ):
        """
        Create a new multistate sampler simulation.

        Parameters
        ----------
        thermodynamic_states : List[ThermodynamicState]
            List of ThermodynamicStates to simulate, with one replica per state.
        sampler_states : List[SamplerState]
            List of initial SamplerStates. The number of states is the number of replicas.
        nbr_lists : List[PairsBase]
            A list of objects used to efficiently calculate interacting pairs for each sampler state.

        Raises
        ------
        RuntimeError
            If the lengths of `thermodynamic_states` and `sampler_states` are not equal.
        """

        self._online_estimator = None

        from chiron.reporters import MultistateReporter

        # Ensure the number of thermodynamic states matches the number of sampler states
        if len(thermodynamic_states) != len(sampler_states):
            raise RuntimeError(
                "Number of thermodynamic states and sampler states must be equal."
            )

        self._allocate_variables(thermodynamic_states, sampler_states, nbr_lists)
        self._reporter = MultistateReporter()

    def _allocate_variables(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        nbr_lists: List[PairsBase],
    ) -> None:
        """
        Allocate and initialize internal variables for the sampler.

        Parameters
        ----------
        thermodynamic_states : List[ThermodynamicState]
            A list of ThermodynamicState objects to be used in the sampler.
        sampler_states : List[SamplerState]
            A list of SamplerState objects for initializing the sampler.
        nbr_lists : List[PairsBase]
            A list of objects used to efficiently calculate interacting pairs for each sampler state.

        Raises
        ------
        RuntimeError
            If the number of MCMC moves and ThermodynamicStates do not match.
        """
        import copy
        import numpy as np

        self._thermodynamic_states = copy.deepcopy(thermodynamic_states)
        self._sampler_states = copy.deepcopy(sampler_states)
        self._nbr_lists = copy.deepcopy(nbr_lists)

        assert len(self._thermodynamic_states) == len(self._sampler_states)
        assert len(self._thermodynamic_states) == len(self._nbr_lists)

        # initial build of neighborlists
        for nbr_list, state in zip(self._nbr_lists, self._sampler_states):
            nbr_list.build(state.positions, state.box_vectors)

        self._replica_thermodynamic_states = np.arange(
            len(thermodynamic_states), dtype=int
        )

        # Initialize matrices for tracking acceptance and proposal statistics.
        self._n_accepted_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._n_proposed_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._energy_thermodynamic_states = np.zeros(
            [self.n_replicas, self.n_states], np.float64
        )
        self._traj = [[] for _ in range(self.n_replicas)]

        # Ensure there is an MCMCSampler for each thermodynamic state.
        from chiron.mcmc import MCMCSampler

        if isinstance(self._mcmc_sampler, MCMCSampler):
            self._mcmc_sampler = [
                copy.deepcopy(self._mcmc_sampler) for _ in range(self.n_states)
            ]
        elif len(self._mcmc_sampler) != self.n_states:
            raise RuntimeError(
                f"The number of MCMCMoves ({len(self._mcmc_sampler)}) and ThermodynamicStates ({self.n_states}) must be the same."
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
            The energy tolerance for minimization (default: 1.0 kJ/mol/nm).
        max_iterations : int, optional
            Maximum number of minimization iterations (default: 1000).

        Notes
        -----
        The minimization modifies the SamplerState associated with the replica.
        """

        from chiron.minimze import minimize_energy
        from loguru import logger as log

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
            sampler_state.positions,
            thermodynamic_state.potential.compute_energy,
            self._nbr_lists[replica_id],
            maxiter=max_iterations,
        )

        # Update the sampler state
        self._sampler_states[replica_id].positions = minimized_state.params

        # it is not likely that we would need to rebuild after minimization
        # but we should make sure check to make sure
        if self._nbr_lists[replica_id].check(
            self._sampler_states[replica_id].positions
        ):
            self._nbr_lists[replica_id].build(
                self._sampler_states[replica_id].positions,
                self._sampler_states[replica_id].box_vectors,
            )

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
        Propagate the state of a single replica using its assigned MCMC move.

        Parameters
        ----------
        replica_id : int
            The index of the replica to propagate.

        Raises
        ------
        RuntimeError
            If an error occurs during the propagation of the replica.
        """

        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        sampler_state = self._sampler_states[replica_id]
        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        nbr_list = self._nbr_lists[replica_id]

        mcmc_sampler = self._mcmc_sampler[thermodynamic_state_id]
        # Propagate using the mcmc sampler
        # NOTE this needs to be updated to support neighborlists
        (
            self._sampler_states[replica_id],
            self._thermodynamic_states[thermodynamic_state_id],
            self._nbr_lists[replica_id],
        ) = mcmc_sampler.run(
            sampler_state, thermodynamic_state, self.number_of_iterations, nbr_list
        )
        # Append the new state to the trajectory for analysis.
        self._traj[replica_id].append(self._sampler_states[replica_id].positions)

    def _perform_swap_proposals(self):
        """
        Perform swap proposals between replicas.

        This method should be overridden by subclasses to implement specific swapping algorithms.

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
        Propose and execute swaps between replicas to enhance sampling efficiency.

        This method handles the logic for proposing swaps between different thermodynamic states
        of the replicas. The specifics of the swapping algorithm depend on subclass implementations.

        Returns
        -------
        np.ndarray
            An array of updated thermodynamic state indices for each replica after swapping.
        """
        from loguru import logger as log

        log.debug("Mixing replicas (does nothing for MultiStateSampler)...")

        # Reset swap attempt counters for this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform the swap proposals and acceptances.
        new_replica_states = self._perform_swap_proposals()

        # Calculate swap acceptance statistics
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        swap_fraction_accepted = (
            n_swaps_accepted / n_swaps_proposed if n_swaps_proposed > 0 else 0.0
        )
        log.debug(
            f"Accepted {n_swaps_accepted}/{n_swaps_proposed} attempted swaps ({swap_fraction_accepted * 100.0:.1f}%)"
        )
        return new_replica_states

    def _propagate_replicas(self) -> None:
        """
        Propagate all replicas through their respective MCMC moves.

        This method applies the corresponding MCMC move to each replica based on its
        current thermodynamic state, thus advancing the state of each replica.
        """
        from loguru import logger as log

        log.debug("Propagating all replicas...")

        # Iterate over all replicas and propagate each one.
        for replica_id in range(self.n_replicas):
            self._propagate_replica(replica_id)

    def _compute_energies(self) -> None:
        """
        Compute the energies of all replicas at all thermodynamic states.

        This method calculates the energy for each replica in every thermodynamic state.
        The energies are stored in the internal energy matrix of the sampler.
        """
        from loguru import logger as log

        log.debug("Computing energy matrix for all replicas...")
        # Initialize the energy matrix and neighborhoods
        self._energy_thermodynamic_states = np.zeros((self.n_replicas, self.n_states))

        # Calculate and store energies for each replica.
        for replica_id in range(self.n_replicas):
            self._energy_thermodynamic_states[
                replica_id, :
            ] = self._compute_replica_energies(replica_id)

    def _is_completed(self, iteration_limit: Optional[int] = None) -> bool:
        """
        Determine if the sampling process has met its completion criteria.

        Checks if the simulation has reached a specified iteration limit or any other
        predefined stopping condition.

        Parameters
        ----------
        iteration_limit : Optional[int], default=None
            An optional iteration limit to check against the current iteration number.

        Returns
        -------
        bool
            True if the simulation has completed based on the stopping criteria, False otherwise.
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
        Execute the replica-exchange simulation for a specified number of iterations.

        Runs the simulation, performing replica propagation, mixing, and energy computation
        for the specified number of iterations.

        Parameters
        ----------
        n_iterations : int, default=10
            The number of iterations to run the simulation.
        """
        from loguru import logger as log

        log.info("Running simulation...")

        self.number_of_iterations = n_iterations

        if self._iteration == 0:
            # Initialize energies if this is the first iteration
            self._compute_energies()
            self._report_iteration()

        # start the sampling loop
        while not self._is_completed(n_iterations):
            self._iteration += 1
            log.info("-" * 80)
            log.info(f"Iteration {self._iteration}/{n_iterations}")
            log.info("-" * 80)

            self._mix_replicas()
            self._propagate_replicas()
            self._compute_energies()
            self._report_iteration()
            self._update_analysis()

        self._reporter.flush_buffer()

    def _report_energy_matrix(self):
        """
        Report the energy matrix for each thermodynamic state.

        This method logs the energy per thermodynamic state, which is useful for analysis
        and debugging purposes.
        """
        from loguru import logger as log

        log.debug("Reporting energy per thermodynamic state...")
        # NOTE: self._energy_thermodynamic_states is transposed from
        # shape (n_replicas, n_states) to (n_states, n_replicas)
        return {"u_kn": self._energy_thermodynamic_states.T}

    def _report_positions(self):
        """
        Store and report the positions of all replicas at the current iteration.

        This method compiles and reports the position data for each replica, which
        is critical for trajectory analysis.
        """
        from loguru import logger as log

        log.debug("Reporting positions...")
        # numpy array with shape (n_replicas, n_atoms, 3)
        xyz = np.zeros((self.n_replicas, self._sampler_states[0].positions.shape[0], 3))
        for replica_id in range(self.n_replicas):
            xyz[replica_id] = self._sampler_states[replica_id].positions
        return {"positions": xyz}

    def _report(self, property: str) -> None:
        """
        Report a specific property of the simulation.

        Depending on the specified property, this method delegates to the appropriate
        internal reporting method.

        Parameters
        ----------
        property : str
            The property to report. Can be 'positions', 'states', 'energies',
            'trajectory', 'mixing_statistics', or 'all'.
        """
        from loguru import logger as log

        log.debug(f"Reporting {property}...")
        if property == "positions":
            return self._report_positions()
        elif property == "states":
            pass
        elif property == "u_kn":
            return self._report_energy_matrix()
        elif property == "trajectory":
            return
        elif "mixing_statistics":
            return

        # match isn't in python 3.9; we can discuss if we want to drop python 3.0 support or just keep the if/else structure
        # match property:
        #     case "positions":
        #         return self._report_positions()
        #     case "states":
        #         pass
        #     case "u_kn":
        #         return self._report_energy_matrix()
        #     case "trajectory":
        #         return
        #     case "mixing_statistics":
        #         return

    def _report_iteration(self):
        """
        Store and report various properties of the current iteration.

        This method is called at each iteration to report essential simulation data,
        such as positions, states, energies, and other properties defined in the reporter.
        """
        from loguru import logger as log

        log.debug("Reporting data for current iteration...")
        log.debug(self._reporter.properties_to_report)
        prop = {}
        for property in self._reporter.properties_to_report:
            p = self._report(property)
            if p:
                prop.update(p)
        self._reporter.report(prop)

    def _update_analysis(self):
        """
        Update the analysis of free energies based on the current simulation data.

        This method is responsible for updating the free energy estimates, either using
        online or offline estimation methods, as configured in the sampler.
        """
        from loguru import logger as log

        log.debug("Updating free energy analysis...")

        # Perform offline free energy estimate if requested
        if self._offline_estimator:
            log.debug("Performing offline free energy estimate...")
            N_k = [self._iteration] * self.n_states
            u_kn = self._reporter.get_property("u_kn")
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
    def f_k(self) -> np.ndarray:
        """
        Get the current free energy estimates.

        Returns the free energy estimates calculated by the sampler's free energy estimator.
        The specific estimator used (online or offline) depends on the sampler configuration.

        Returns
        -------
        np.ndarray
            Array of free energy estimates for each thermodynamic state.

        Raises
        ------
        RuntimeError
            If no free energy estimator is found.
        """

        if self._offline_estimator:
            return self._offline_estimator.f_k
        elif self._online_estimator:
            return self._online_estimator.f_k
        else:
            raise RuntimeError("No free energy estimator found.")
