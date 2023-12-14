# This file contains various routines related to generating pair lists

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional
from .states import SamplerState
from loguru import logger as log
from openmm import unit


# split out the displacement calculation from the neighborlist for flexibility
from abc import ABC, abstractmethod
class Space(ABC):
    def __init__(self, box_vectors: Optional[jnp.array]=None) -> None:
        if box_vectors is not None:
            self.box_vectors = box_vectors
    @property
    def box_vectors(self) -> jnp.array:
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, box_vectors: jnp.array) -> None:
        self._box_vectors = box_vectors

    @abstractmethod
    def displacement(self, xyz_1: jnp.array, xyz_2: jnp.array) -> Tuple[jnp.array, jnp.array]:
        pass

    @abstractmethod
    def wrap(self, xyz: jnp.array) -> jnp.array:
        pass

class OrthogonalPeriodicSpace(Space):
    """
    Calculate the periodic distance between two points.

    Returns
    -------
    Callable
        Function that calculates the periodic displacement and distance between two points
    """
    def __init__(self, box_vectors: Optional[jnp.array]=None) -> None:
        super().__init__(box_vectors)
        if box_vectors is not None:
            self.box_lengths = jnp.array([box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]])

    @property
    def box_vectors(self) -> jnp.array:
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, box_vectors: jnp.array) -> None:
        self._box_vectors = box_vectors
        self._box_lengths = jnp.array([box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]])

    @partial(jax.jit, static_argnums=(0,))
    def displacement(self, xyz_1: jnp.array, xyz_2: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Coordinates of the first point
        xyz_2: jnp.array
            Coordinates of the second point

        Returns
        -------
        r_ij: jnp.array
            Displacement vector between the two points
        dist: float
            Distance between the two points

        """
        # calculate uncorrect r_ij
        r_ij = xyz_1 - xyz_2

                # calculated corrected displacement vector
        r_ij = (
            jnp.mod(r_ij + self._box_lengths * 0.5, self._box_lengths)
            - self._box_lengths * 0.5
        )
        # calculate the scalar distance
        dist = jnp.linalg.norm(r_ij, axis=-1)

        return r_ij, dist
    @partial(jax.jit, static_argnums=(0,))
    def wrap(self, xyz: jnp.array) -> jnp.array:
        """
        Wrap the coordinates of the system.

        Parameters
        ----------
        xyz: jnp.array
            Coordinates of the system

        Returns
        -------
        jnp.array
            Wrapped coordinates of the system

        """
        xyz = xyz - jnp.floor(xyz/self._box_lengths) * self._box_lengths

        return xyz
class OrthogonalNonperiodicSpace(Space):

    @partial(jax.jit, static_argnums=(0,))
    def displacement(self, xyz_1: jnp.array, xyz_2: jnp.array, ) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Coordinates of the first point
        xyz_2: jnp.array
            Coordinates of the second point

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

    @partial(jax.jit, static_argnums=(0,))
    def wrap(self, xyz: jnp.array) -> jnp.array:
        """
        Wrap the coordinates of the system.
        For the Non-periodic system, this does not alter the coordinates

        Parameters
        ----------
        xyz: jnp.array
            Coordinates of the system

        Returns
        -------
        jnp.array
            Wrapped coordinates of the system

        """
        return xyz

class NeighborListNsqrd:
    """
    N^2 neighborlist implementation that returns the particle pair ids, displacement vectors, and distances.

    Parameters
    ----------
    space: Space
        Class that defines how to calculate the displacement between two points and apply the boundary conditions
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
        space: Space,
        cutoff: unit.Quantity = unit.Quantity(1.2, unit.nanometer),
        skin: unit.Quantity = unit.Quantity(0.4, unit.nanometer),
        n_max_neighbors: float=200,
    ):
        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)
        self.skin = skin.value_in_unit_system(unit.md_unit_system)
        self.cutoff_and_skin = self.cutoff + self.skin
        self.n_max_neighbors = n_max_neighbors
        self.space = space

        # set a a simple variable to know if this has at least been built once as opposed to just initialized
        # this does not imply that the neighborlist is up to date
        self.is_built= False

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
        r_ij, dist = self.space.displacement(particle_i, coordinates)

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
        # the call to x0 and box_vectors automatically convert these to jnp arrays in the correct unit system
        self.ref_coordinates = sampler_state.x0
        self.box_vectors = sampler_state.box_vectors

        # the neighborlist assumes that the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list
        self.space.box_vectors = self.box_vectors

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
            log.debug(
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

        self.is_built = True

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
        r_ij, dist = self.space.displacement(
            coordinates[particles1], coordinates[neighbors]
        )
        # calculate the mask to determine if the particle is a neighbor
        # this will be done based on the interaction cutoff and using the neighbor_mask to exclude padding
        mask = jnp.where((dist < self.cutoff) & (neighbor_mask), 1, 0)

        # calculate the number of pairs
        n_pairs = mask.sum()

        return n_pairs, mask, dist, r_ij

    def calculate(self, coordinates:jnp.array):
        """
        Calculate the neighbor list for the current state

        Parameters
        ----------
        coordinates: jnp.array
            Shape[N,3] array of particle coordinates

        Returns
        -------
        n_neighbors: jnp.array
            Array of number of neighbors for each particle
        padding_mask: jnp.array
            Array of masks to exclude padding from the neighbor list of each particle
        dist: jnp.array
            Array of distances between each particle and its neighbors
        r_ij: jnp.array
            Array of displacement vectors between each particle and its neighbors
        """
        #coordinates = sampler_state.x0
        #note, we assume the box vectors do not change between building and calculating the neighbor list
        #changes to the box vectors require rebuilding the neighbor list

        n_neighbors, padding_mask, dist, r_ij = jax.vmap(
            self._calc_distance_per_particle, in_axes=(0, 0, 0, None)
        )(self.particle_ids, self.neighbor_list, self.neighbor_mask, coordinates)
        # mask = mask.reshape(-1, self.n_max_neighbors)
        return n_neighbors, padding_mask, dist, r_ij

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

        r_ij, displacement = self.space.displacement(
            coordinates[particle], ref_coordinates[particle]
        )

        status = jnp.where(displacement >= self.skin / 2.0, True, False)
        del displacement
        return status

    def check(self, coordinates:jnp.array)->bool:
        """
        Check if the neighbor list needs to be rebuilt based on displacement of the particles from the reference coordinates.
        If a particle moves more than 0.5 skin distance, the neighborlist will be rebuilt.

        Note, this could also accept a user defined criteria for distance, but this is not implemented yet.

        Parameters
        ----------
        coordinates: jnp.array
            Array of particle coordinates
        Returns
        -------
        bool
            True if the neighbor list needs to be rebuilt, False if it does not.
        """
        status = jax.vmap(
            self._calculate_particle_displacement, in_axes=(0, None, None)
        )(self.particle_ids, coordinates, self.ref_coordinates)
        if jnp.any(status):
            del status
            return True
        else:
            del status
            return False
