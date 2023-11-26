"""Markov chain Monte Carlo simulation framework.

This module provides a framework for equilibrium sampling from a given
thermodynamic state of a biomolecule using a Markov chain Monte Carlo scheme.

It currently offer supports for
* Langevin dynamics,
* Monte Carlo,
* Monte Carlo barostat moves,

which can be combined through the SequenceMove classes.

>>> from chiron import unit
>>> from openmmtools.testsystems import AlanineDipeptideVacuum
>>> from chiron.states import StateVariablesCollection
>>> from chiron.potential import NeuralNetworkPotential
>>> from modelforge.potential.pretrained_models import SchNetModel
>>> from chiron.mcmc import MCMCSampler, SequenceMove, MCMove, LangevinDynamicsMove
 
Create the initial state for an alanine
dipeptide system in vacuum.

>>> alanine_dipeptide = AlanineDipeptideVacuum()
>>> potential = NeuralNetworkPotential(SchNetModel, alanine_dipeptide.topology)
>>> state = StateVariablesCollection(temperature=298*unit.kelvin, positions=test.positions)

Create an MCMC move to sample the equilibrium distribution.

>>> langevin_move = LangevinDynamicsMove(n_steps=10)

>>> mc_move = MCMove(timestep=1.0*unit.femtosecond, n_steps=50)
>>> sampler = MCMCSampler(state, move=ghmc_move)

You can combine them to form a sequence of moves

>>> sequence_move = SequenceMove([ghmc_move, langevin_move])
>>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=sequence_move)

"""


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

    def __init__(self, state_variables, move_set):
        from copy import deepcopy

        # Make a deep copy of the state so that initial state is unchanged.
        self.state_variables = deepcopy(state_variables)
        self.move = move_set

    def run(self, n_iterations=1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.

        """
        # Apply move for n_iterations.
