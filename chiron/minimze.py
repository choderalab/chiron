import jax
import jax.numpy as jnp
from jaxopt import GradientDescent

def minimize_energy(coordinates, potential_fn, nbr_list=None, maxiter=1000):
    """
    Minimize the potential energy of a system using JAXopt.

    Parameters
    ----------
    coordinates : jnp.ndarray
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

    def objective_fn(x):
        if nbr_list is not None:
            return potential_fn(x, nbr_list)
        else:
            return potential_fn(x)

    optimizer = GradientDescent(
        fun=jax.value_and_grad(objective_fn), value_and_grad=True, maxiter=maxiter
    )
    result = optimizer.run(coordinates)

    return result.params
