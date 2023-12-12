# This file contains various routines related to generating pair lists

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Callable
from .states import SamplerState
from loguru import logger as log

# split out the displacement calculation from the neighborlist for flexibility

DisplacementFn = Callable[jnp.array, jnp.array, jnp.array]
def orthogonal_periodic_system(periodicity: Tuple[bool, bool, bool]=[True, True, True]) -> DisplacementFn:
    """
    Calculate the periodic distance between two points.

    Parameters
    ----------
    periodicity: Tuple[bool, bool, bool]
        Periodicity of the system

    Returns
    -------
    Callable
        Function that calculates the periodic displacement and distance between two points
    """
    box_mask = jnp.array(periodicity).astype(int)

    @jax.jit
    def displacement_fn(xyz_1: jnp.array, xyz_2: jnp.array, box_vec: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Coordinates of the first point
        xyz_2: jnp.array
            Coordinates of the second point
        box_vec: jnp.array
            Box vector of the system

        Returns
        -------
        r_ij: jnp.array
            Displacement vector between the two points
        dist: float
            Distance between the two points

        """
        # calculate uncorrect r_ij
        r_ij = xyz_1 - xyz_2

        # 0 if dimension i is not periodic, box_vec[i] if periodic
        box_vec_periodicity = box_vec * box_mask

        # calculated corrected displacement vector
        r_ij = (
            jnp.mod(r_ij + box_vec_periodicity * 0.5, box_vec)
            - box_vec_periodicity * 0.5
        )
        # calculate the scalar distance
        dist = jnp.linalg.norm(r_ij, axis=-1)

        return r_ij, dist

    return displacement_fn

def orthogonal_nonperiodic_system() -> DisplacementFn:
    """
    Calculate the periodic distance between two points.

    Parameters
    ----------
    p
    Returns
    -------
    Callable
        Function that calculates the  displacement and distance between two points
    """

    @jax.jit
    def displacement_fn(xyz_1: jnp.array, xyz_2: jnp.array, box_vec: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Coordinates of the first point
        xyz_2: jnp.array
            Coordinates of the second point
        box_vec: jnp.array
            Box vector of the system

        Returns
        -------
        r_ij: jnp.array
            Displacement vector between the two points
        dist: float
            Distance between the two points

        """
        # calculate uncorrect r_ij
        r_ij = xyz_1 - xyz_2

        # calculate the scalar distance
        dist = jnp.linalg.norm(r_ij, axis=-1)

        return r_ij, dist

    return displacement_fn

class NeighborListNsqrd:
    """
    N^2 neighborlist implementation that returns the particle pair ids, displacement vectors, and distances.

    Parameters
    ----------
    cutoff: float, default = 2.5
        Cutoff distance for the neighborlist
    skin: float, default = 0.4
        Skin distance for the neighborlist
    n_max_neighbors: int, default=200
        Maximum number of neighbors for each particle.  Used for padding arrays for efficient jax computations
        This will be checked and dynamically updated during the build stage
    Examples
    --------


    """

    def __init__(
        self,
        displacement_fn: DisplacementFn,
        cutoff: float = 2.5,
        skin:float=0.4,
        n_max_neighbors=200,
    ):
        self.cutoff = cutoff
        self.skin = skin
        self.cutoff_and_skin = self.cutoff + self.skin
        self.n_max_neighbors = n_max_neighbors
        self.displacement_fn = displacement_fn

    # note, we need to use the partial decorator in order to use the jit decorate
    # so that it knows to ignore the `self` argument
    @partial(jax.jit, static_argnums=(0,))
    def _pairs_mask(self, particle_ids: jnp.array):
        """
        Jitted function to generate mask that allows us to remove self-interactions and double-counting of pairs

        Parameters
        ----------
        particle_ids: jnp.array
            Array of particle ids

        Returns
        -------
        jnp.array
            Bool mask to remove self-interactions and double-counting of pairs

        """
        # for the nsq approach, we consider the distance between a particle and all other particles in the system
        # if we used a cell list the possible_neighbors would be a smaller list, i.e., only those in the neigboring cells

        possible_neighbors = particle_ids

        particles_j = jnp.broadcast_to(
            possible_neighbors,
            (particle_ids.shape[0], possible_neighbors.shape[0]),
        )

        # reshape the particle_ids
        particles_i = jnp.reshape(particle_ids, (particle_ids.shape[0], 1))
        # create a mask to exclude self interactions and double counting
        temp_mask = particles_i < particles_j

        return temp_mask

    @partial(jax.jit, static_argnums=(0, 4))
    def _build_neighborlist(
        self, particle_i, reduction_mask, coordinates, n_max_neighbors
    ):
        """
        Jitted function to build the neighbor list for a single particle

        Parameters
        ----------
        particle_i: jnp.array
            X,Y,Z coordinates of particle i
        reduction_mask: jnp.array
            Mask to exclude self-interactions and double counting of pairs
        coordinates: jnp.array
            X,Y,Z coordinates of all particles
        n_max_neighbors: int
            Maximum number of neighbors for each particle.  Used for padding arrays for efficient jax computations

        Returns
        -------
        neighbor_list_mask: jnp.array
            Mask to exclude padding from the neighbor list
        neighbor_list: jnp.array
            List of particle ids for the neighbors, padded to n_max_neighbors
        n_neighbors: int
            Number of neighbors for the particle
        """

        # calculate the displacement between particle i and all other particles
        r_ij, dist = self.displacement_fn(particle_i, coordinates, self.box_vec)

        # neighbor_mask will be an array of length n_particles (i.e., length of coordinates)
        # where each element is True if the particle is a neighbor, False if it is not
        # subject to both the cutoff+skin and the reduction mask that eliminates double counting and self-interactions
        neighbor_mask = jnp.where(
            (dist < self.cutoff_and_skin) & (reduction_mask), True, False
        )
        # when we  pad the neighbor list, we will use last particle id in the neighbor list
        # this choice was made such that when we use the neighbor list in the masked energy calculat
        # the padded values will result in reasonably well defined values
        fill_value = jnp.argmax(neighbor_mask)

        # count up the number of neighbors
        n_neighbors = jnp.where(neighbor_mask, 1, 0).sum()

        # since neighbor_mask indices have a one-to-one correspondence to particle ids,
        # applying jnp.where, will return an array of the indices that are neighbors.
        # since this needs to be uniformly sized, we can just fill this array up to the n_max_neighbors.
        neighbor_list = jnp.array(
            jnp.where(neighbor_mask, size=n_max_neighbors, fill_value=fill_value),
            dtype=jnp.uint16,
        )
        # we need to generate a new mask associatd with the padded neighbor list
        # to be able to quickly exclude the padded values from the neighbor list
        neighbor_list_mask = jnp.where(jnp.arange(self.n_max_neighbors) < n_neighbors, 1, 0)

        del r_ij, dist
        return neighbor_list_mask, neighbor_list, n_neighbors

    def build(self, sampler_state: SamplerState):
        # set our reference coordinates
        self.ref_coordinates = sampler_state.x0
        self.box_vec = sampler_state.box_vectors

        # store the ids of all the particles
        self.particle_ids = jnp.array(range(0, self.ref_coordinates.shape[0]), dtype=jnp.uint16)

        # calculate which pairs to exclude
        reduction_mask = self._pairs_mask(self.particle_ids)

        # calculate the distance for all pairs this will return
        # neighbor_mask: an array of shape (n_particles, n_particles) where each element is the mask
        # to determine if the particle is a neighbor
        # neighbor_list: an array of shape (n_particles, n_max_neighbors) where each element is the particle id of the neighbor
        # this is padded with zeros to ensure a uniform size;
        # n_neighbors: an array of shape (n_particles) where each element is the number of neighbors for that particle

        self.neighbor_mask, self.neighbor_list, self.n_neighbors = jax.vmap(
            self._build_neighborlist, in_axes=(0, 0, None, None)
        )(
            self.ref_coordinates,
            reduction_mask,
            self.ref_coordinates,
            self.n_max_neighbors,
        )

        self.neighbor_list = self.neighbor_list.reshape(-1, self.n_max_neighbors)

        if jnp.any(self.n_neighbors == self.n_max_neighbors).block_until_ready():
            self.n_max_neighbors = int(jnp.max(self.n_neighbors) + 10)
            print(
                f"Increasing n_max_neighbors from {self.n_max_neighbors} to at  {jnp.max(self.n_neighbors)+10}"
            )
            self.neighbor_mask, self.neighbor_list, self.n_neighbors = jax.vmap(
                self._build_neighborlist, in_axes=(0, 0, None, None)
            )(
                self.ref_coordinates,
                reduction_mask,
                self.ref_coordinates,
                self.n_max_neighbors,
            )

            self.neighbor_list = self.neighbor_list.reshape(-1, self.n_max_neighbors)

    @partial(jax.jit, static_argnums=(0,))
    def _calc_distance_per_particle(
        self, particle1, neighbors, neighbor_mask, coordinates
    ):
        """
        Jitted function to calculate the distance between a particle and its neighbors

        Parameters
        ----------
        particle1: int
            Particle id
        neighbors: jnp.array
            Array of particle ids for the neighbors of particle1
        neighbor_mask: jnp.array
            Mask to exclude padding from the neighbor list of particle1
        coordinates: jnp.array
            X,Y,Z coordinates of all particles

        Returns
        -------
        n_pairs: int
            Number of interacting pairs for the particle
        mask: jnp.array
            Mask to exclude padding from the neighbor list of particle1.
            If a particle is within the interaction cutoff, the mask is 1, otherwise it is 0
        dist: jnp.array
            Array of distances between the particle and its neighbors
        r_ij: jnp.array
            Array of displacement vectors between the particle and its neighbors
        """
        # repeat the particle id for each neighbor
        particles1 = jnp.repeat(particle1, neighbors.shape[0])

        # calculate the displacement between particle i and all  neighbors
        r_ij, dist = self.displacement_fn(
            coordinates[particles1], coordinates[neighbors], self.box_vec
        )
        # calculate the mask to determine if the particle is a neighbor
        # this will be done based on the interaction cutoff and using the neighbor_mask to exclude padding
        mask = jnp.where((dist < self.cutoff) & (neighbor_mask), 1, 0)

        # calculate the number of pairs
        n_pairs = mask.sum()

        return n_pairs, mask, dist, r_ij

    def calculate(self, sampler_state: SamplerState):
        """
        Calculate the neighbor list for the current state

        Parameters
        ----------
        sampler_state: SamplerState
            Sampler state object

        Returns
        -------
        n_neighbors: jnp.array
            Array of number of neighbors for each particle
        mask: jnp.array
            Array of masks to exclude padding from the neighbor list of each particle
        dist_full: jnp.array
            Array of distances between each particle and its neighbors
        r_ij_full: jnp.array

        """
        coordinates = sampler_state.x0

        n_neighbors, mask, dist_full, r_ij_full = jax.vmap(
            self._calc_distance_per_particle, in_axes=(0, 0, 0, None)
        )(self.particle_ids, self.neighbor_list, self.neighbor_mask, coordinates)
        # mask = mask.reshape(-1, self.n_max_neighbors)
        return n_neighbors, mask, dist_full, r_ij_full

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_particle_displacement(self, particle, coordinates, ref_coordinates):
        """
        Calculate the displacement of a particle from the reference coordinates.
        If the displacement exceeds the half the skin distance, return True, otherwise return False.

        This function is designed to allow it to be jitted and vmapped over particle indices.

        Parameters
        ----------
        particle: int
            Particle id
        coordinates: jnp.array
            Array of particle coordinates
        ref_coordinates: jnp.array
            Array of reference particle coordinates

        Returns
        -------
        bool
            True if the particle is outside the skin distance, False if it is not.
        """
        # calculate the displacement of a particle from the initial coordinates
        # r_ij = coordinates[particle] - ref_coordinates[particle]
        # r_ij = jnp.mod(r_ij + self.box_mask * 0.5, self.box_vec) - self.box_mask * 0.5
        r_ij, displacement = self.displacement_fn(
            coordinates[particle], ref_coordinates[particle], self.box_vec
        )

        status = jnp.where(displacement >= self.skin / 2.0, True, False)
        del displacement
        return status

    def check(self, sampler_state: SamplerState):
        """
        Check if the neighbor list needs to be rebuilt.

        Returns
        -------
        bool
            True if the neighbor list needs to be rebuilt, False if it does not.
        """
        coordinates = sampler_state.x0
        status = jax.vmap(
            self._calculate_particle_displacement, in_axes=(0, None, None)
        )(self.particle_ids, coordinates, self.ref_coordinates)
        if jnp.any(status):
            del status
            return True
        else:
            del status
            return False
