import os
import copy
import time
from typing import List
import loguru as logger
from chiron.states import SamplerState, ThermodynamicState
import datetime
from loguru import logger as log
import numpy as np
from openmmtools.utils import time_it

import openmm
from openmm import unit


class MultiStateSampler(object):
    """
    Base class for samplers that sample multiple thermodynamic states using
    one or more replicas.

    This base class provides a general simulation facility for multistate from multiple
    thermodynamic states, allowing any set of thermodynamic states to be specified.
    If instantiated on its own, the thermodynamic state indices associated with each
    state are specified and replica mixing does not change any thermodynamic states,
    meaning that each replica remains in its original thermodynamic state.

    Stored configurations, energies, swaps, and restart information are all written
    to a single output file using the platform portable, robust, and efficient
    NetCDF4 library.

    Parameters
    ----------
    mcmc_moves : MCMCMove or list of MCMCMove, optional
        The MCMCMove used to propagate the thermodynamic states. If a list of MCMCMoves,
        they will be assigned to the correspondent thermodynamic state on
        creation. If None is provided, Langevin dynamics with 2fm timestep, 5.0/ps collision rate,
        and 500 steps per iteration will be used.
    number_of_iterations : int or infinity, optional, default: 1
        The number of iterations to perform. Both ``float('inf')`` and
        ``numpy.inf`` are accepted for infinity. If you set this to infinity,
        be sure to set also ``online_analysis_interval``.
    online_analysis_interval : None or Int >= 1, optional, default: 200
        Choose the interval at which to perform online analysis of the free energy.

        After every interval, the simulation will be stopped and the free energy estimated.

        If the error in the free energy estimate is at or below ``online_analysis_target_error``, then the simulation
        will be considered completed.

        If set to ``None``, then no online analysis is performed

    online_analysis_target_error : float >= 0, optional, default 0.0
        The target error for the online analysis measured in kT per phase.

        Once the free energy is at or below this value, the phase will be considered complete.

        If ``online_analysis_interval`` is None, this option does nothing.

        Default is set to 0.0 since online analysis runs by default, but a finite ``number_of_iterations`` should also
        be set to ensure there is some stop condition. If target error is 0 and an infinite number of iterations is set,
        then the sampler will run until the user stop it manually.

    online_analysis_minimum_iterations : int >= 0, optional, default 200
        Set the minimum number of iterations which must pass before online analysis is carried out.

        Since the initial samples likely not to yield a good estimate of free energy, save time and just skip them
        If ``online_analysis_interval`` is None, this does nothing

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

    def __init__(
        self,
        mcmc_moves=None,
        number_of_iterations=1,
    ):
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

    def create(
        self,
        thermodynamic_states: List[ThermodynamicState],
        sampler_states: List[SamplerState],
        metadata=None,
    ):
        """Create new multistate sampler simulation.

        Parameters
        ----------
        thermodynamic_states : list of ThermodynamicState
            Thermodynamic states to simulate, where one replica is allocated per state.
            Each state must have a system with the same number of atoms.
        sampler_states : list of SamplerState
            One or more sets of initial sampler states.
            The number of replicas is taken to be the number of sampler states provided.
            If the sampler states do not have box_vectors attached and the system is periodic,
            an exception will be thrown.
        metadata : dict, optional, default=None
           Simulation metadata to be stored in the file.
        """
        # TODO: initialize reporter here
        # TODO: consider unsampled thermodynamic states for reweighting schemes
        self._allocate_variables(thermodynamic_states, sampler_states)

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

    def _allocate_variables(self, thermodynamic_states, sampler_states):
        # Save thermodynamic states. This sets n_replicas.
        self._thermodynamic_states = [
            copy.deepcopy(thermodynamic_state)
            for thermodynamic_state in thermodynamic_states
        ]

        # Deep copy sampler states.
        self._sampler_states = [
            copy.deepcopy(sampler_state) for sampler_state in sampler_states
        ]

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
        self._n_accepted_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._n_proposed_matrix = np.zeros([self.n_states, self.n_states], np.int64)

        # Allocate memory for energy matrix. energy_thermodynamic_states[k][l]
        # is the reduced potential computed at the positions of SamplerState sampler_states[k]
        # and ThermodynamicState thermodynamic_states[l].
        self._energy_thermodynamic_states = np.zeros(
            [self.n_replicas, self.n_states], np.float64
        )
        self._neighborhoods = np.zeros([self.n_replicas, self.n_states], "i1")

    def _minimize_replica(
        self, replica_id: int, tolerance: unit.Quantity, max_iterations: int
    ):
        from chiron.minimze import minimize_energy

        # Retrieve thermodynamic and sampler states.
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        sampler_state = self._sampler_states[replica_id]

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.get_reduced_potential(sampler_state)
        print(initial_energy)
        log.debug(
            f"Replica {replica_id + 1}/{self.n_replicas}: initial energy {initial_energy:8.3f}kT"
        )

        results = minimize_energy(
            sampler_state.x0, lj_potential.compute_energy, nbr_list, maxiter=0
        )

    def minimize(
        self,
        tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometers,
        max_iterations=0,
    ):
        """Minimize all replicas.

        Minimized positions are stored at the end.

        Parameters
        ----------
        tolerance : openmm.unit.Quantity, optional
            Minimization tolerance (units of energy/mole/length, default is
            ``1.0 * unit.kilojoules_per_mole / unit.nanometers``).
        max_iterations : int, optional
            Maximum number of iterations for minimization. If 0, minimization
            continues until converged.

        """
        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError(
                "Cannot minimize replicas. The simulation must be created first."
            )

        log.debug("Minimizing all replicas...")

        # minimization
        minimized_positions, sampler_state_ids = [], []
        for replica_id in range(self.n_replicas):
            minimized_position, sampler_state_id = self._minimize_replica(
                replica_id, tolerance, max_iterations
            )
            minimized_positions.append(minimized_position)
            sampler_state_ids.append(sampler_state_id)

        # Update all sampler states.
        for sampler_state_id, minimized_pos in zip(
            sampler_state_ids, minimized_positions
        ):
            self._sampler_states[sampler_state_id].positions = minimized_pos

    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate all replicas.

        This does not increase the iteration counter. The equilibrated
        positions are stored at the end.

        Parameters
        ----------
        n_iterations : int
            Number of equilibration iterations.
        mcmc_moves : MCMCMove or list of MCMCMove, optional
            Optionally, the MCMCMoves to use for equilibration can be
            different from the ones used in production.

        """
        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError(
                "Cannot equilibrate replicas. The simulation must be created first."
            )

        # If no MCMCMove is specified, use the ones for production.
        if mcmc_moves is None:
            mcmc_moves = self._mcmc_moves

        # Make sure there is one MCMCMove per thermodynamic state.
        if isinstance(mcmc_moves, mcmc.MCMCMove):
            mcmc_moves = [copy.deepcopy(mcmc_moves) for _ in range(self.n_states)]
        elif len(mcmc_moves) != self.n_states:
            raise RuntimeError(
                "The number of MCMCMoves ({}) and ThermodynamicStates ({}) for equilibration"
                " must be the same.".format(len(self._mcmc_moves), self.n_states)
            )
        from openmmtools.utils import Timer

        timer = Timer()
        timer.start("Run Equilibration")

        # Temporarily set the equilibration MCMCMoves.
        production_mcmc_moves = self._mcmc_moves
        self._mcmc_moves = mcmc_moves
        for iteration in range(1, 1 + n_iterations):
            logger.info("Equilibration iteration {}/{}".format(iteration, n_iterations))
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
        timer.report_timing()

        # Restore production MCMCMoves.
        self._mcmc_moves = production_mcmc_moves

        # TODO: Update stored positions.

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
        if self._iteration == 0:
            try:
                self._compute_energies()
            # We're intercepting a possible initial NaN position here thrown by OpenMM, which is a simple exception
            # So we have to under-specify this trap.
            except Exception as e:
                if "coordinate is nan" in str(e).lower():
                    err_message = "Initial coordinates were NaN! Check your inputs!"
                    logger.critical(err_message)
                    raise SimulationNaNError(err_message)
                else:
                    # If not the special case, raise the error normally
                    raise e
            mpiplus.run_single_node(
                0,
                self._reporter.write_energies,
                self._energy_thermodynamic_states,
                self._neighborhoods,
                self._energy_unsampled_states,
                self._iteration,
            )
            self._check_nan_energy()

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

            logger.info("*" * 80)
            logger.info("Iteration {}/{}".format(self._iteration, iteration_limit))
            logger.info("*" * 80)
            timer.start("Iteration")

            # Update thermodynamic states
            self._replica_thermodynamic_states = self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Write iteration to storage file
            self._report_iteration()

            # Update analysis
            self._update_analysis()

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
            logger.info(
                "Iteration took {:.3f}s.".format(self._timing_data["iteration_seconds"])
            )
            if self._timing_data["estimated_time_remaining"] != float("inf"):
                logger.info(
                    "Estimated completion in {}, at {} (consuming total wall clock time {}).".format(
                        self._timing_data["estimated_time_remaining"],
                        self._timing_data["estimated_localtime_finish_date"],
                        self._timing_data["estimated_total_time"],
                    )
                )

            # Perform sanity checks to see if we should terminate here.
            self._check_nan_energy()
