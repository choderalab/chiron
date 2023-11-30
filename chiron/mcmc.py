"""Markov chain Monte Carlo simulation framework.

This module provides a framework for equilibrium sampling from a given
thermodynamic state of a biomolecule using a Markov chain Monte Carlo scheme.

It currently offer supports for
* Langevin dynamics,
* Monte Carlo,

which can be combined through the SequenceMove classes.

>>> from chiron import unit
>>> from openmmtools.testsystems import AlanineDipeptideVacuum
>>> from chiron.states import SimulationVariables
>>> from chiron.potential import NeuralNetworkPotential
>>> from modelforge.potential.pretrained_models import SchNetModel
>>> from chiron.mcmc import MCMCSampler, SequenceMove, MCMove, LangevinDynamicsMove
 
Create the initial state for an alanine
dipeptide system in vacuum.

>>> alanine_dipeptide = AlanineDipeptideVacuum()
>>> potential = NeuralNetworkPotential(SchNetModel, alanine_dipeptide.topology)
>>> state = SimulationVariables(temperature=298*unit.kelvin, positions=test.positions)

Create an MCMC move to sample the equilibrium distribution.

>>> langevin_move = LangevinDynamicsMove(n_steps=10)

>>> mc_move = MCMove(timestep=1.0*unit.femtosecond, n_steps=50)
>>> sampler = MCMCSampler(state, move=ghmc_move)

You can combine them to form a sequence of moves

>>> sequence_move = SequenceMove([ghmc_move, langevin_move])
>>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=sequence_move)

"""
from chiron.states import SamplerState, ThermodynamicState
from chiron.potential import NeuralNetworkPotential
from openmm import unit
from loguru import logger as log
from typing import Dict, Union, Tuple, List
import jax.numpy as jnp

from typing import Optional


class StateUpdateMove:
    def __init__(self):
        """
        Initialize the MCMove with a molecular system.

        """
        # system represents the potential energy function and topology
        pass


class LangevinDynamicsMove(StateUpdateMove):
    def __init__(
        self,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
    ):
        """
        Initialize the LangevinDynamicsMove with a molecular system.

        Parameters
        ----------
        stepsize : unit.Quantity
            Time step size for the integration.
        collision_rate : unit.Quantity
            Collision rate for the Langevin dynamics.
        """
        self.stepsize = stepsize
        self.collision_rate = collision_rate

        from chiron.integrators import LangevinIntegrator

        self.integrator = LangevinIntegrator(
            stepsize=self.stepsize,
            collision_rate=self.collision_rate,
        )

    def run(
        self,
        potential: NeuralNetworkPotential,
        x0: unit.Quantity,
        state_variables: SimulationState,
        n_steps: int,
    ):
        """
        Run the integrator to perform molecular dynamics simulation.

        Args:
            state_variables (StateVariablesCollection): State variables of the system.
        """

        self.integrator.run(
            x0=x0,
            potential=potential,
            box_vectors=state_variables.box_vectors,
            temperature=state_variables.temperature,
            n_steps=n_steps,
        )


class MCMove(StateUpdateMove):
    def __init__(self) -> None:
        self.system: Optional[NeuralNetworkPotential] = None

    def _initialize_system(self, state_variables: SimulationState):
        if self.system is None:
            self.system = self.NeuralNetworkPotential(state_variables)

    def _check_state_compatiblity(
        self,
        old_state: SimulationState,
        new_state: SimulationState,
    ):
        """
        Check if the states are compatible.

        Parameters
        ----------
        old_state : StateVariablesCollection
            The state of the system before the move.
        new_state : StateVariablesCollection
            The state of the system after the move.

        Raises
        ------
        ValueError
            If the states are not compatible.
        """
        pass

    def apply_move(self):
        """
        Apply a Monte Carlo move to the system.

        This method should be overridden by subclasses to define specific types of moves.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in subclasses.
        """

        raise NotImplementedError("apply_move() must be implemented in subclasses")

    def compute_acceptance_probability(
        self,
        old_state: SimulationState,
        new_state: SimulationState,
    ):
        """
        Compute the acceptance probability for a move from an old state to a new state.

        Parameters
        ----------
        old_state : object
            The state of the system before the move.
        new_state : object
            The state of the system after the move.

        Returns
        -------
        float
            Acceptance probability as a float.
        """
        self._check_state_compatiblity(old_state, new_state)
        old_system = self.system(old_state)
        new_system = self.system(new_state)

        energy_before_state_change = old_system.compute_energy(old_state.position)
        enegy_after_state_change = new_system.compute_energy(new_state.position)
        # Implement the logic to compute the acceptance probability
        pass

    def accept_or_reject(self, probability):
        """
        Decide whether to accept or reject the move based on the acceptance probability.

        Parameters
        ----------
        probability : float
            Acceptance probability.

        Returns
        -------
        bool
            Boolean indicating if the move is accepted.
        """
        import jax.numpy as jnp

        return jnp.random.rand() < probability


class RotamerMove(MCMove):
    def apply_move(self):
        """
        Implement the logic specific to rotamer changes.
        """
        pass


class ProtonationStateMove(MCMove):
    def apply_move(self):
        """
        Implement the logic specific to protonation state changes.
        """
        pass


class TautomericStateMove(MCMove):
    def apply_move(self):
        """
        Implement the logic specific to tautomeric state changes.
        """
        pass


class ProposedPositionMove(MCMove):
    def apply_move(self):
        """
        Implement the logic specific to a MC position change.
        """
        pass


class MoveSet:
    """
    Represents a set of moves for a Markov Chain Monte Carlo (MCMC) algorithm.
    """

    def __init__(
        self,
        available_moves: Dict[str, StateUpdateMove],
        move_schedule: List[Tuple[str, int]],
    ) -> None:
        """
        Initializes a MoveSet object.

        Parameters
        ----------
        available_moves : Dict[str, StateUpdateMove]
            A dictionary of available moves, where the keys are move names and the values are StateUpdateMove objects.
        move_sequence : List[Tuple[str, int]]
            A list representing the move sequence, where each tuple contains a move name and an integer representing the number of times the move should be performed in sequence.

        Raises
        ------
        ValueError
            If a move in the sequence is not present in available_moves.
        """
        self.available_moves = available_moves
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
        for move_name, _ in self.move_schedule:
            if move_name not in self.available_moves:
                raise ValueError(f"Move {move_name} in the sequence is not available.")

    def add_move(self, new_moves: Dict[str, MCMove]):
        """
        Adds new moves to the available moves.

        Parameters
        ----------
        new_moves : Dict[str, MCMove]
            A dictionary of new moves to be added, where the keys are move names and the values are MCMove objects.
        """
        self.available_moves.update(new_moves)

    def remove_move(self, move_name: str):
        """
        Removes a move from the available moves.

        Parameters
        ----------
        move_name : str
            The name of the move to be removed.
        """
        del self.available_moves[move_name]


class GibbsSampler(object):
    """Basic Markov chain Monte Carlo Gibbs sampler.

    Parameters
    ----------
    StateVariablesCollection : states.StateVariablesCollection
        Defines the states describing the conditional distributions.
    move_set : container of MarkovChainMonteCarloMove objects
        Moves to attempt during MCMC run.
        The move set can be a single move or a sequence of moves.
        The moves will define the joint distributions that are sampled.

    """

    def __init__(
        self, state_variables: Union[SimulationState, Dict], move_set: MoveSet
    ):
        from copy import deepcopy

        log.info("Initializing Gibbs sampler")

        # Make a deep copy of the state so that initial state is unchanged.
        self.state_variables = deepcopy(state_variables)
        self.move = move_set

    def run(self, x0: unit.Quantity):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        xO : unit.Quantity
            Initial positions of the particles.

        """
        log.info("Running Gibbs sampler")
        log.info(f"move_set = {self.move.available_moves}")
        log.info(f"move_schedule = {self.move.move_schedule}")
        for move, n_steps in self.move.move_schedule:
            self.move.available_moves[move].run(x0, self.state_variables, n_steps)


class MetropolizedMove:
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

    def __init__(self, atom_subset=None, **kwargs):
        super(MetropolizedMove, self).__init__(**kwargs)
        self.n_accepted = 0
        self.n_proposed = 0
        self.atom_subset = atom_subset

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value["n_accepted"]
        self.n_proposed = value["n_proposed"]

    def apply(self, thermodynamic_state, sampler_state):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state to apply the move to. This is modified.

        """
        import copy
        import jax.numpy as jnp

        # Compute initial energy
        initial_energy = thermodynamic_state.reduced_potential(sampler_state)
        # Store initial positions of the atoms that are moved.
        # We'll use this also to recover in case the move is rejected.
        atom_subset = self.atom_subset
        if isinstance(atom_subset, slice):
            # Numpy array when sliced return a view, they are not copied.
            initial_positions = copy.deepcopy(sampler_state.positions[atom_subset])
        else:
            # This automatically creates a copy.
            initial_positions = sampler_state.positions[atom_subset]

        # Propose perturbed positions. Modifying the reference changes the sampler state.
        proposed_positions = self._propose_positions(initial_positions)

        # Compute the energy of the proposed positions.
        sampler_state.positions[atom_subset] = proposed_positions

        proposed_energy = thermodynamic_state.reduced_potential(sampler_state)

        # Accept or reject with Metropolis criteria.
        delta_energy = proposed_energy - initial_energy
        if not jnp.isnan(proposed_energy) and (
            delta_energy <= 0.0 or jnp.random.rand() < jnp.exp(-delta_energy)
        ):
            self.n_accepted += 1
        else:
            # Restore original positions.
            sampler_state.positions[atom_subset] = initial_positions
        self.n_proposed += 1

    def _propose_positions(self, positions: jnp.ndarray):
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
