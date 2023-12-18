import jax
import jax.numpy as jnp
from jaxopt import GradientDescent


def minimize_energy(coordinates, potential_fn, nbr_list=None):
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

    Returns
    -------
    jnp.ndarray
        The optimized coordinates.
    """

    def objective_fn(x):
        x_reshaped = x.reshape(coordinates.shape)
        if nbr_list is not None:
            return potential_fn(x_reshaped, nbr_list)
        else:
            return potential_fn(x_reshaped)

    optimizer = GradientDescent(fun=jax.value_and_grad(objective_fn), maxiter=500)
    result = optimizer.run(jnp.array(coordinates.flatten()))

    return result.params.reshape(coordinates.shape)
