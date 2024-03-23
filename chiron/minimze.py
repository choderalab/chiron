from typing import Callable
from jax import numpy as jnp


def minimize_energy(
    coordinates: jnp.array, potential_fn: Callable, nbr_list=None, maxiter: int = 1000
):
    """
    Minimize the potential energy of a system using JAXopt.

    Parameters
    ----------
    coordinates : jnp.array
        The initial coordinates of the system.
    potential_fn : callable
        The potential energy function of the system, which takes coordinates as input.
    nbr_list : NeighborList, optional
        The neighbor list object (if required by the potential function).
    maxiter: int, optional
        The maximum number of iterations to run the minimizer.

    Returns
    -------
    jnp.ndarray
        The optimized coordinates.
    """
    from loguru import logger as log

    def objective_fn(x):
        if nbr_list is not None:
            log.debug("Using neighbor list")
            return potential_fn(x, nbr_list)
        else:
            log.debug("Using NO neighbor list")
            return potential_fn(x)

    from jaxopt import GradientDescent
    import jax

    optimizer = GradientDescent(
        fun=jax.value_and_grad(objective_fn), value_and_grad=True, maxiter=maxiter
    )
    result = optimizer.run(coordinates)

    return result
