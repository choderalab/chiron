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
from chiron.states import SimulationState
from chiron.potential import NeuralNetworkPotential
from openmm import unit
from loguru import logger as log


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

    def __init__(self, state_variables: SimulationState, move_set: MoveSet):
        from copy import deepcopy

        log.info("Initializing Gibbs sampler")

        # Make a deep copy of the state so that initial state is unchanged.
        self.state_variables = deepcopy(state_variables)
        self.move = move_set

    def run(self, n_iterations: int = 1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.

        """
        # Apply move for n_iterations.


from typing import Optional


class LangevinDynamicsMove:
    def __init__(
        self,
        n_steps: int,
        NeuralNetworkPotential: NeuralNetworkPotential,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
    ):
        """
        Initialize the LangevinDynamicsMove with a molecular system.

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to perform.
        NeuralNetworkPotential : object
            A representation of the molecular system (neural network potential and topology).
        stepsize : unit.Quantity
            Time step size for the integration.
        collision_rate : unit.Quantity
            Collision rate for the Langevin dynamics.
        """

        self.n_steps = n_steps
        self.NeuralNetworkPotential = NeuralNetworkPotential
        self.stepsize = stepsize
        self.collision_rate = collision_rate
        # system represents the potential energy function and topology
        self.system: Optional[NeuralNetworkPotential] = None

    def run(
        self,
        state_variables: SimulationState,
    ):
        """
        Run the integrator to perform molecular dynamics simulation.

        Args:
            state_variables (StateVariablesCollection): State variables of the system.
        """
        from chiron.integrator import LangevinIntegrator

        if self.system is None:
            self.system = self.NeuralNetworkPotential(state_variables)

        dynamics = LangevinIntegrator(
            self.system, self.system.topology, state_variables.box_vectors
        )

        dynamics.run(
            x0=state_variables.positions,
            temperature=state_variables.temperature,
            n_steps=self.n_steps,
            stepsize=self.stepsize,
            collision_rate=self.collision_rate,
        )


class MoveSet:
    def __init__(self) -> None:
        pass


class MCMove:
    def __init__(self, NeuralNetworkPotential: NeuralNetworkPotential):
        """
        Initialize the MCMove with a molecular system.

        Parameters
        ----------
        system : object
            A representation of the molecular system (e.g., coordinates, topology).
        """
        self.system = NeuralNetworkPotential

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
        old_system = self.NeuronNetworkPotential(old_state)
        new_system = self.NeuronNetworkPotential(new_state).compute_energy(
            new_state.position
        )

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
