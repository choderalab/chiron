class GibbsSampler(object):
    """Basic Markov chain Monte Carlo Gibbs sampler.

    Parameters
    ----------
    conditional_state : states.ConditionalDistribution
        Initial conditional distribution.
    move_set : container of MarkovChainMonteCarloMove objects
        Moves to attempt during MCMC run. If list or tuple, will run all moves each
        iteration in specified sequence (e.g. [move1, move2, move3]). If dict, will
        use specified unnormalized weights (e.g. { move1 : 0.3, move2 : 0.5, move3, 0.9 })

    Examples
    --------

    Create and run an alanine dipeptide simulation with a weighted move.

    """


    def __init__(self, conditional_state, move_set):
        from copy import deepcopy

        # Make a deep copy of the state so that initial state is unchanged.
        self.conditional_state = deepcopy(conditional_state)
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
