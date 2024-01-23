from chiron.states import SamplerState, ThermodynamicState
from openmm import unit
from typing import Tuple, List, Optional
import jax.numpy as jnp
from chiron.reporters import LangevinDynamicsReporter, _SimulationReporter

from abc import ABC, abstractmethod


class MCMCMove:
    def __init__(
        self,
        nr_of_moves: int,
        reporter: Optional[_SimulationReporter] = None,
        report_frequency: Optional[int] = 100,
    ):
        """
        Initialize a move within the molecular system.

        Parameters
        ----------
        nr_of_moves : int
            Number of moves to be applied.
        reporter : _SimulationReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_frequency : int, optional
        """

        self.nr_of_moves = nr_of_moves
        self.reporter = reporter
        self.report_frequency = report_frequency
        from loguru import logger as log

        if self.reporter is not None:
            log.info(
                f"Using reporter {self.reporter} saving to {self.reporter.workdir}"
            )
            assert self.report_frequency is not None

    @abstractmethod
    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Update the state of the system.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state to run the integrator on.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to run the integrator on.
        nbr_list : PairsBase, optional
            The neighbor list to use for the simulation.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state.
        thermodynamic_state : ThermodynamicState
            The updated thermodynamic state.

        """
        pass


class LangevinDynamicsMove(MCMCMove):
    def __init__(
        self,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        reporter: Optional[LangevinDynamicsReporter] = None,
        report_frequency: int = 100,
        nr_of_steps=1_000,
        save_traj_in_memory: bool = False,
    ):
        """
        Initialize the LangevinDynamicsMove with a molecular system.

        Parameters
        ----------
        stepsize : unit.Quantity
            Time step size for the integration.
        collision_rate : unit.Quantity
            Collision rate for the Langevin dynamics.
        reporter : LangevinDynamicsReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_frequency : int
            Frequency of saving the simulation data.
            Default is 100.
        nr_of_steps : int, optional
            Number of steps to run the integrator for.
            Default is 1_000.
        save_traj_in_memory: bool
            Flag indicating whether to save the trajectory in memory.
            Default is False. NOTE: Only for debugging purposes.
        """
        super().__init__(
            nr_of_moves=nr_of_steps,
            reporter=reporter,
            report_frequency=report_frequency,
        )

        self.stepsize = stepsize
        self.collision_rate = collision_rate
        self.save_traj_in_memory = save_traj_in_memory
        self.traj = []
        from chiron.integrators import LangevinIntegrator

        self.integrator = LangevinIntegrator(
            stepsize=self.stepsize,
            collision_rate=self.collision_rate,
            report_frequency=report_frequency,
            reporter=reporter,
            save_traj_in_memory=save_traj_in_memory,
        )

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Run the integrator to perform molecular dynamics simulation.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state to run the integrator on.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to run the integrator on.
        nbr_list : PairsBase, optional
            The neighbor list to use for the simulation.
        """

        assert isinstance(
            sampler_state, SamplerState
        ), f"Sampler state must be SamplerState, not {type(sampler_state)}"
        assert isinstance(
            thermodynamic_state, ThermodynamicState
        ), f"Thermodynamic state must be ThermodynamicState, not {type(thermodynamic_state)}"

        updated_sampler_state = self.integrator.run(
            thermodynamic_state=thermodynamic_state,
            sampler_state=sampler_state,
            n_steps=self.nr_of_moves,
            nbr_list=nbr_list,
        )

        if self.save_traj_in_memory:
            self.traj.append(self.integrator.traj)
            self.integrator.traj = []

        # The thermodynamic_state will not change for the langevin move
        return updated_sampler_state, thermodynamic_state


class MCMove(MCMCMove):
    def __init__(
        self,
        nr_of_moves: int,
        reporter: Optional[_SimulationReporter],
        report_frequency: int = 1,
        method: str = "metropolis",
    ) -> None:
        """
        Initialize the move.

        Parameters
        ----------
        nr_of_moves
            Number of moves to be applied in each call to update.
        reporter
            Reporter object for saving the simulation step data.
        report_frequency
            Frequency of saving the simulation data.
        method
            Methodology to use for accepting or rejecting the proposed state.
            Default is "metropolis".
        """
        super().__init__(nr_of_moves, reporter=reporter)
        self.method = "metropolis"  # I think we should pass a class/function instead of a string, like space.

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Perform the defined move and update the state.

        Parameters
        ----------
        sampler_state : SamplerState
            The initial state of the simulation, including positions.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system, including temperature and potential.
        nbr_list : PairBase, optional
            Neighbor list for the system.


        """
        calculate_current_potential = True
        for i in range(self.nr_of_moves):
            self._step(
                sampler_state,
                thermodynamic_state,
                nbr_list,
                calculate_current_potential=calculate_current_potential,
            )
            # after the first step, we don't need to recalculate the current potential, it will be stored
            calculate_current_potential = False

    def _step(
        self,
    ):
        # if this is the first time we are calling this,
        # we will need to recalculate the reduced potential for the current state
        if calculate_current_potential:
            current_reduced_pot = current_thermodynamic_state.get_reduced_potential(
                current_sampler_state
            )
            # save the current_reduced_pot so we don't have to recalculate
            # it on the next iteration if the move is rejected
            self._current_reduced_pot = current_reduced_pot
        else:
            current_reduced_pot = self._current_reduced_pot

        # propose a new state and calculate the log proposal ratio
        # this will be specific to the type of move
        # in addition to the sampler_state, this will require/return the thermodynamic state
        # for systems that e.g., make changes to particle identity.
        (
            proposed_sampler_state,
            log_proposal_ratio,
            proposed_thermodynamic_state,
        ) = self._propose(current_sampler_state, current_thermodynamic_state)

        # calculate the reduced potential for the proposed state
        proposed_reduced_pot = proposed_thermodynamic_state.get_reduced_potential(
            proposed_sampler_state
        )

        # accept or reject the proposed state
        decision = self._accept_or_reject(
            current_reduced_pot,
            proposed_reduced_pot,
            log_proposal_ratio,
            method=self.method,
        )

        if decision:
            # save the reduced potential of the accepted state so
            # we don't have to recalculate it the next iteration
            self._current_reduced_pot = proposed_reduced_pot

            # replace the current state with the proposed state
            # not sure this needs to be a separate function but for simplicity in outlining the code it is fine
            # or should this return the new sampler_state and thermodynamic_state?
            self._replace_states(
                current_sampler_state,
                proposed_sampler_state,
                current_thermodynamic_state,
                proposed_thermodynamic_state,
            )

        # a function that will update the statistics for the move
        self._update_statistics(decision)

    @abstractmethod
    def _propose(self, current_sampler_state, current_thermodynamic_state):
        """
        Propose a new state and calculate the log proposal ratio.

        This will need to be defined for each move

        Parameters
        ----------
        current_sampler_state : SamplerState, required
            Current sampler state.
        current_thermodynamic_state : ThermodynamicState, required

        Returns
        -------
        proposed_sampler_state : SamplerState
            Proposed sampler state.
        log_proposal_ratio : float
            Log proposal ratio.
        proposed_thermodynamic_state : ThermodynamicState
            Proposed thermodynamic state.

        """
        pass

    def _replace_states(
        self,
        current_sampler_state,
        proposed_sampler_state,
        current_thermodynamic_state,
        proposed_thermodynamic_state,
    ):
        """
        Replace the current state with the proposed state.
        """
        # define the code to copy the proposed state to the current state

    def _accept_or_reject(
        self,
        current_reduced_pot,
        proposed_reduced_pot,
        log_proposal_ratio,
        method=method,
    ):
        """
        Accept or reject the proposed state with a given methodology.
        """
        # define the acceptance probability


class RotamerMove(MCMove):
    def _step(self):
        """
        Implement the logic specific to rotamer changes.
        """
        pass


class ProtonationStateMove(MCMove):
    def _step(self):
        """
        Implement the logic specific to protonation state changes.
        """
        pass


class TautomericStateMove(MCMove):
    def _step(self):
        """
        Implement the logic specific to tautomeric state changes.
        """
        pass


class MoveSchedule:
    """
    Represents an (optimizable) series of moves for a Markov Chain Monte Carlo (MCMC) algorithm.

    Parameters
    ----------
    move_schedule : List[Tuple[str, StateUpdateMove]]
        A list representing the move schedule, where each tuple contains a move name and a move instance.

    Raises
    ------
    ValueError
        If a move in the schedule is not an instance of StateUpdateMove.
    """

    def __init__(
        self,
        move_schedule: List[Tuple[str, MCMCMove]],
    ) -> None:
        _AVAILABLE_MOVES = ["LangevinDynamicsMove"]
        self.move_schedule = move_schedule

        self._validate_sequence()

    def _validate_sequence(self):
        """
        Validates the move sequence against the available moves.

        Raises
        ------
        ValueError
            If a move in the sequence is not present in available_moves.
        """
        for move_name, move_class in self.move_schedule:
            if not isinstance(move_class, MCMCMove):
                raise ValueError(f"Move {move_name} in the sequence is not available.")


class MCMCSampler:
    """
    Basic Markov chain Monte Carlo sampler.

    Parameters
    ----------
    move_set : MoveSet
        Set of moves to attempt during MCMC run.
    sampler_state : SamplerState
        Initial sampler state.
    thermodynamic_state : ThermodynamicState
        Thermodynamic state describing the system.

    Examples
    """

    def __init__(
        self,
        move_set: MoveSchedule,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
    ):
        from copy import deepcopy
        from loguru import logger as log

        log.info("Initializing Gibbs sampler")
        self.move = move_set
        self.sampler_state = deepcopy(sampler_state)
        self.thermodynamic_state = deepcopy(thermodynamic_state)

    def run(self, n_iterations: int = 1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int, optional
            Number of iterations of the sampler to run.
        """
        from loguru import logger as log

        log.info("Running MCMC sampler")
        log.info(f"move_schedule = {self.move.move_schedule}")
        for iteration in range(n_iterations):
            log.info(f"Iteration {iteration + 1}/{n_iterations}")
            for move_name, move in self.move.move_schedule:
                log.debug(f"Performing: {move_name}")
                move.run(self.sampler_state, self.thermodynamic_state)

        log.info("Finished running MCMC sampler")
        log.debug("Closing reporter")
        for _, move in self.move.move_schedule:
            if move.reporter is not None:
                move.reporter.flush_buffer()
                # TODO: flush reporter
                log.debug(f"Closed reporter {move.reporter.log_file_path}")


from .neighbors import PairsBase


class MetropolizedMove(MCMove):
    """A base class for metropolized moves.

    Only the proposal needs to be specified by subclasses through the method
    _propose_positions().

    Parameters
    ----------
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).

    Attributes
    ----------
    n_accepted : int
        The number of proposals accepted.
    n_proposed : int
        The total number of attempted moves.
    atom_subset

    Examples
    --------
    TBC
    """

    def __init__(
        self,
        atom_subset: Optional[List[int]] = None,
        nr_of_moves: int = 100,
        reporter: Optional[_SimulationReporter] = None,
        report_frequency: int = 1,
    ):
        self.n_accepted = 0
        self.n_proposed = 0
        self.atom_subset = atom_subset
        super().__init__(nr_of_moves=nr_of_moves, reporter=reporter)
        from loguru import logger as log

        self.report_frequency = report_frequency
        log.debug(f"Atom subset is {atom_subset}.")

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value["n_accepted"]
        self.n_proposed = value["n_proposed"]

    def apply(
        self,
        thermodynamic_state: ThermodynamicState,
        sampler_state: SamplerState,
        nbr_list=Optional[PairsBase],
    ):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : SamplerState
           The initial sampler state to apply the move to. This is modified.
        nbr_list: Neighbor List or Pair List routine,
            The routine to use to calculate the interacting atoms.
            Default is None and will use an unoptimized pairlist without PBC
        """
        import jax.numpy as jnp
        from loguru import logger as log

        # Compute initial energy
        initial_energy = thermodynamic_state.get_reduced_potential(
            sampler_state, nbr_list
        )  # NOTE: in kT
        log.debug(f"Initial energy is {initial_energy} kT.")

        # Store initial positions of the atoms that are moved.
        x0 = sampler_state.x0
        atom_subset = self.atom_subset
        if atom_subset is None:
            initial_positions = jnp.copy(x0)
        else:
            initial_positions = jnp.copy(sampler_state.x0[jnp.array(atom_subset)])
        log.debug(f"Initial positions are {initial_positions} nm.")
        # Propose perturbed positions. Modifying the reference changes the sampler state.
        proposed_positions = self._propose_positions(initial_positions)

        log.debug(f"Proposed positions are {proposed_positions} nm.")
        # Compute the energy of the proposed positions.
        if atom_subset is None:
            sampler_state.x0 = proposed_positions
        else:
            sampler_state.x0 = sampler_state.x0.at[jnp.array(atom_subset)].set(
                proposed_positions
            )
        if nbr_list is not None:
            if nbr_list.check(sampler_state.x0):
                nbr_list.build(sampler_state.x0, sampler_state.box_vectors)

        proposed_energy = thermodynamic_state.get_reduced_potential(
            sampler_state, nbr_list
        )  # NOTE: in kT
        # Accept or reject with Metropolis criteria.
        delta_energy = proposed_energy - initial_energy
        log.debug(f"Delta energy is {delta_energy} kT.")
        import jax.random as jrandom

        self.key, subkey = jrandom.split(self.key)

        compare_to = jrandom.uniform(subkey)
        if not jnp.isnan(proposed_energy) and (
            delta_energy <= 0.0 or compare_to < jnp.exp(-delta_energy)
        ):
            self.n_accepted += 1
            log.debug(f"Check suceeded: {compare_to=}  < {jnp.exp(-delta_energy)}")
            log.debug(
                f"Move accepted. Energy change: {delta_energy:.3f} kT. Number of accepted moves: {self.n_accepted}."
            )
            if self.n_proposed % self.report_frequency == 0:
                self.reporter.report(
                    {
                        "energy": proposed_energy,  # in kT
                        "step": self.n_proposed,
                        "traj": sampler_state.x0,
                    }
                )
        else:
            # Restore original positions.
            if atom_subset is None:
                sampler_state.x0 = initial_positions
            else:
                sampler_state.x0 = sampler_state.x0.at[jnp.array([atom_subset])].set(
                    initial_positions
                )
            log.debug(
                f"Move rejected. Energy change: {delta_energy:.3f} kT. Number of rejected moves: {self.n_proposed - self.n_accepted}."
            )
        self.n_proposed += 1

    def _propose_positions(self, positions: jnp.array):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 jnp.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 jnp.ndarray
            The new proposed positions.

        """
        raise NotImplementedError(
            "This MetropolizedMove does not know how to propose new positions."
        )


class MetropolisDisplacementMove(MetropolizedMove):
    """A metropolized move that randomly displace a subset of atoms.

    Parameters
    ----------
    displacement_sigma : openmm.unit.Quantity
        The standard deviation of the normal distribution used to propose the
        random displacement (units of length, default is 1.0*nanometer).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).

    Attributes
    ----------
    n_accepted : int
        The number of proposals accepted.
    n_proposed : int
        The total number of attempted moves.
    displacement_sigma
    atom_subset

    See Also
    --------
    MetropolizedMove

    """

    def __init__(
        self,
        displacement_sigma=1.0 * unit.nanometer,
        nr_of_moves: int = 100,
        atom_subset: Optional[List[int]] = None,
        reporter: Optional[LangevinDynamicsReporter] = None,
    ):
        """
        Initialize the MCMC class.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator. Default is 1234.
        displacement_sigma : float or unit.Quantity, optional
            The standard deviation of the displacement for each move. Default is 1.0 nm.
        nr_of_moves : int, optional
            The number of moves to perform. Default is 100.
        atom_subset : list of int, optional
            A subset of atom indices to consider for the moves. Default is None.
        reporter : SimulationReporter, optional
            The reporter to write the data to. Default is None.
        Returns
        -------
        None
        """
        super().__init__(nr_of_moves=nr_of_moves, reporter=reporter)
        self.displacement_sigma = displacement_sigma
        self.atom_subset = atom_subset
        self.key = None

    def displace_positions(
        self, positions: jnp.array, displacement_sigma=1.0 * unit.nanometer
    ):
        """Return the positions after applying a random displacement to them.

        Parameters
        ----------
        positions : nx3 jnp.array unit.Quantity
            The positions to displace.
        displacement_sigma : openmm.unit.Quantity
            The standard deviation of the normal distribution used to propose
            the random displacement (units of length, default is 1.0*nanometer).

        Returns
        -------
        rotated_positions : nx3 numpy.ndarray openmm.unit.Quantity
            The displaced positions.

        """
        import jax.random as jrandom

        self.key, subkey = jrandom.split(self.key)
        nr_of_atoms = positions.shape[0]
        unitless_displacement_sigma = displacement_sigma.value_in_unit_system(
            unit.md_unit_system
        )
        displacement_vector = (
            jrandom.normal(subkey, shape=(nr_of_atoms, 3)) * 0.1
        )  # NOTE: convert from Angstrom to nm
        scaled_displacement_vector = displacement_vector * unitless_displacement_sigma
        updated_position = positions + scaled_displacement_vector

        return updated_position

    def _propose_positions(self, initial_positions: jnp.array) -> jnp.array:
        """Implement MetropolizedMove._propose_positions for apply()."""
        return self.displace_positions(initial_positions, self.displacement_sigma)

    def run(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list=None,
        progress_bar=True,
    ):
        from tqdm import tqdm
        from loguru import logger as log
        from jax import random

        self.key = sampler_state.new_PRNG_key

        for trials in (
            tqdm(range(self.nr_of_moves)) if progress_bar else range(self.nr_of_moves)
        ):
            self.apply(thermodynamic_state, sampler_state, nbr_list)
            if trials % 100 == 0:
                log.debug(f"Acceptance rate: {self.n_accepted / self.n_proposed}")
                if self.reporter is not None:
                    self.reporter.report(
                        {
                            "Acceptance rate": self.n_accepted / self.n_proposed,
                            "step": self.n_proposed,
                        }
                    )

        log.info(f"Acceptance rate: {self.n_accepted / self.n_proposed}")
