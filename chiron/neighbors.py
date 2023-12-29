# This file contains various routines related to generating pair lists

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, Union
from .states import SamplerState
from loguru import logger as log
from openmm import unit


# split out the displacement calculation from the neighborlist for flexibility
from abc import ABC, abstractmethod


class Space(ABC):
    def __init__(
        self, box_vectors: Union[jnp.array, unit.Quantity, None] = None
    ) -> None:
        """
        Abstract base class for defining the simulation space.

        Parameters
        ----------
        box_vectors: jnp.array, optional
            Box vectors for the system.
        """
        if box_vectors is not None:
            if isinstance(box_vectors, unit.Quantity):
                if not box_vectors.unit.is_compatible(unit.nanometer):
                    raise ValueError(
                        f"Box vectors require distance unit, not {box_vectors.unit}"
                    )
                self.box_vectors = box_vectors.value_in_unit_system(unit.md_unit_system)
            elif isinstance(box_vectors, jnp.ndarray):
                if box_vectors.shape != (3, 3):
                    raise ValueError(
                        f"box_vectors should be a 3x3 array, shape provided: {box_vectors.shape}"
                    )

                self.box_vectors = box_vectors
            else:
                raise TypeError(
                    f"box_vectors must be a jnp.array or unit.Quantity, not {type(box_vectors)}"
                )

    @property
    def box_vectors(self) -> jnp.array:
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, box_vectors: jnp.array) -> None:
        self._box_vectors = box_vectors

    @abstractmethod
    def displacement(
        self, xyz_1: jnp.array, xyz_2: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        pass

    @abstractmethod
    def wrap(self, xyz: jnp.array) -> jnp.array:
        pass


class OrthogonalPeriodicSpace(Space):
    """
    Defines the simulation space for an orthogonal periodic system.

    """

    @property
    def box_vectors(self) -> jnp.array:
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, box_vectors: jnp.array) -> None:
        self._box_vectors = box_vectors
        self._box_lengths = jnp.array(
            [box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]]
        )

    @partial(jax.jit, static_argnums=(0,))
    def displacement(
        self, xyz_1: jnp.array, xyz_2: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
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
        xyz = xyz - jnp.floor(xyz / self._box_lengths) * self._box_lengths

        return xyz


class OrthogonalNonperiodicSpace(Space):
    @partial(jax.jit, static_argnums=(0,))
    def displacement(
        self,
        xyz_1: jnp.array,
        xyz_2: jnp.array,
    ) -> Tuple[jnp.array, jnp.array]:
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


class PairsBase(ABC):
    """
    Abstract Base Class for different algorithms that determine which particles are interacting.

    Parameters
    ----------
    space: Space
        Class that defines how to calculate the displacement between two points and apply the boundary conditions
    cutoff: float, default = 2.5
        Cutoff distance for the neighborlist

    Examples
    --------
    >>> from chiron.neighbors import PairsBase, OrthogonalPeriodicSpace
    >>> from chiron.states import SamplerState
    >>> from openmm as unit
    >>> import jax.numpy as jnp
    >>>
    >>> space = OrthogonalPeriodicSpace() # define the simulation space, in this case an orthogonal periodic space
    >>> sampler_state = SamplerState(x0=jnp.array([[0.0, 0.0, 0.0], [2, 0.0, 0.0], [0.0, 2, 0.0]]),
    >>>                              box_vectors=jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]]))
    >>>
    >>> pair_list = PairsBase(space, cutoff=2.5*unit.nanometer) # initialize the pair list
    >>> pair_list.build_from_state(sampler_state) # build the pair list from the sampler state
    >>>
    >>> coordinates = sampler_state.x0 # get the coordinates from the sampler state, without units attached
    >>>
    >>> # the calculate function will produce information used to calculate the energy
    >>> n_neighbors, padding_mask, dist, r_ij = pair_list.calculate(coordinates)
    >>>
    """

    def __init__(
        self,
        space: Space,
        cutoff: unit.Quantity = unit.Quantity(1.2, unit.nanometer),
    ):
        if not isinstance(space, Space):
            raise TypeError(f"space must be of type Space, found {type(space)}")
        if not cutoff.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, cutoff.unit = {cutoff.unit}"
            )
        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)
        self.space = space

    @abstractmethod
    def build(
        self,
        coordinates: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build list from an array of coordinates and array of box vectors.

        Parameters
        ----------
        coordinates: jnp.array or unit.Quantity
            Shape[n_particles,3] array of particle coordinates, either with or without units attached.
            If the array is passed as a unit.Quantity, the units must be distances and will be converted to nanometers.
        box_vectors: jnp.array or unit.Quantity
            Shape[3,3] array of box vectors for the system, either with or without units attached.
            If the array is passed as a unit.Quantity, the units must be distances and will be converted to nanometers.

        Returns
        -------
        None

        """
        pass

    def _validate_build_inputs(
        self,
        coordinates: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Validate the inputs to the build function.
        """
        if isinstance(coordinates, unit.Quantity):
            if not coordinates.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Coordinates require distance units, not {coordinates.unit}"
                )
            self.ref_coordinates = coordinates.value_in_unit_system(unit.md_unit_system)
        if isinstance(coordinates, jnp.ndarray):
            if coordinates.shape[1] != 3:
                raise ValueError(
                    f"coordinates should be a Nx3 array, shape provided: {coordinates.shape}"
                )
            self.ref_coordinates = coordinates
        if isinstance(box_vectors, unit.Quantity):
            if not box_vectors.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Box vectors require distance unit, not {box_vectors.unit}"
                )
            self.box_vectors = box_vectors.value_in_unit_system(unit.md_unit_system)

        if isinstance(box_vectors, jnp.ndarray):
            if box_vectors.shape != (3, 3):
                raise ValueError(
                    f"box_vectors should be a 3x3 array, shape provided: {box_vectors.shape}"
                )
            self.box_vectors = box_vectors

    def build_from_state(self, sampler_state: SamplerState):
        """
        Build the list from a SamplerState object

        Parameters
        ----------
        sampler_state: SamplerState
            SamplerState object containing the coordinates and box vectors

        Returns
        -------
        None
        """
        if not isinstance(sampler_state, SamplerState):
            raise TypeError(f"Expected SamplerState, got {type(sampler_state)} instead")

        coordinates = sampler_state.x0
        if sampler_state.box_vectors is None:
            raise ValueError(f"SamplerState does not contain box vectors")
        box_vectors = sampler_state.box_vectors

        self.build(coordinates, box_vectors)

    @abstractmethod
    def calculate(self, coordinates: jnp.array):
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
        pairs: jnp.array
            Array of particle ids for the possible neighbors of each particle.
            The size of this array will depend on the underlying algorithm.
        padding_mask: jnp.array
            Array of masks to exclude padding from the neighbor list of each particle
        dist: jnp.array
            Array of distances between each particle and its neighbors
        r_ij: jnp.array
            Array of displacement vectors between each particle and its neighbors
        """
        pass

    @abstractmethod
    def check(self, coordinates: jnp.array) -> bool:
        """
        Check if the internal variables need to be reset. E.g., rebuilding a neighborlist
        Should do nothing for a simple pairlist.

        Parameters
        ----------
        coordinates: jnp.array
            Array of particle coordinates
        Returns
        -------
        bool
            True if the neighbor list needs to be rebuilt, False if it does not.
        """
        pass

    @property
    def box_vectors(self) -> jnp.array:
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, box_vectors: jnp.array) -> None:
        self._box_vectors = box_vectors
        self.space.box_vectors = box_vectors


class NeighborListNsqrd(PairsBase):
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
        n_max_neighbors: float = 200,
    ):
        if not isinstance(space, Space):
            raise TypeError(f"space must be of type Space, found {type(space)}")
        if not cutoff.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, cutoff.unit = {cutoff.unit}"
            )
        if not skin.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, skin.unit = {skin.unit}"
            )

        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)
        self.skin = skin.value_in_unit_system(unit.md_unit_system)
        self.cutoff_and_skin = self.cutoff + self.skin
        self.n_max_neighbors = n_max_neighbors
        self.space = space

        # set a a simple variable to know if this has at least been built once as opposed to just initialized
        # this does not imply that the neighborlist is up to date
        self.is_built = False

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

    @partial(jax.jit, static_argnums=(0, 5))
    def _build_neighborlist(
        self, particle_i, reduction_mask, pid, coordinates, n_max_neighbors
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
        fill_value = jnp.where(fill_value == pid, fill_value + 1, fill_value)

        # count up the number of neighbors
        n_neighbors = jnp.where(neighbor_mask, 1, 0).sum()

        # since neighbor_mask indices have a one-to-one correspondence to particle ids,
        # applying jnp.where, will return an array of the indices that are neighbors.
        # since this needs to be uniformly sized, we can just fill this array up to the n_max_neighbors.
        neighbor_list = jnp.array(
            jnp.where(neighbor_mask, size=n_max_neighbors, fill_value=fill_value),
            dtype=jnp.uint32,
        )
        # we need to generate a new mask associatd with the padded neighbor list
        # to be able to quickly exclude the padded values from the neighbor list
        neighbor_list_mask = jnp.where(jnp.arange(n_max_neighbors) < n_neighbors, 1, 0)

        del r_ij, dist
        return neighbor_list_mask, neighbor_list, n_neighbors

    def build(
        self,
        coordinates: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build the neighborlist from an array of coordinates and box vectors.

        Parameters
        ----------
        coordinates: jnp.array
            Shape[N,3] array of particle coordinates
        box_vectors: jnp.array
            Shape[3,3] array of box vectors

        Returns
        -------
        None

        """

        # set our reference coordinates
        # the call to x0 and box_vectors automatically convert these to jnp arrays in the correct unit system
        if isinstance(coordinates, unit.Quantity):
            if not coordinates.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Coordinates require distance units, not {coordinates.unit}"
                )
            coordinates = coordinates.value_in_unit_system(unit.md_unit_system)

        if isinstance(box_vectors, unit.Quantity):
            if not box_vectors.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Box vectors require distance unit, not {box_vectors.unit}"
                )
            box_vectors = box_vectors.value_in_unit_system(unit.md_unit_system)

        if box_vectors.shape != (3, 3):
            raise ValueError(
                f"box_vectors should be a 3x3 array, shape provided: {box_vectors.shape}"
            )

        # will also set the box vectors in the space object due to the setter in the ABC
        self.box_vectors = box_vectors

        self.ref_coordinates = coordinates
        # the neighborlist assumes that the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list
        # self.space.box_vectors = self.box_vectors

        # store the ids of all the particles
        self.particle_ids = jnp.array(
            range(0, self.ref_coordinates.shape[0]), dtype=jnp.uint32
        )

        # calculate which pairs to exclude
        reduction_mask = self._pairs_mask(self.particle_ids)

        # calculate the distance for all pairs this will return
        # neighbor_mask: an array of shape (n_particles, n_particles) where each element is the mask
        # to determine if the particle is a neighbor
        # neighbor_list: an array of shape (n_particles, n_max_neighbors) where each element is the particle id of the neighbor
        # this is padded with zeros to ensure a uniform size;
        # n_neighbors: an array of shape (n_particles) where each element is the number of neighbors for that particle

        self.neighbor_mask, self.neighbor_list, self.n_neighbors = jax.vmap(
            self._build_neighborlist, in_axes=(0, 0, 0, None, None)
        )(
            self.ref_coordinates,
            reduction_mask,
            self.particle_ids,
            self.ref_coordinates,
            self.n_max_neighbors,
        )

        self.neighbor_list = self.neighbor_list.reshape(-1, self.n_max_neighbors)

        while jnp.any(self.n_neighbors == self.n_max_neighbors).block_until_ready():
            log.debug(
                f"Increasing n_max_neighbors from {self.n_max_neighbors} to at  {jnp.max(self.n_neighbors)+10}"
            )
            self.n_max_neighbors = int(jnp.max(self.n_neighbors) + 10)

            self.neighbor_mask, self.neighbor_list, self.n_neighbors = jax.vmap(
                self._build_neighborlist, in_axes=(0, 0, 0, None, None)
            )(
                self.ref_coordinates,
                reduction_mask,
                self.particle_ids,
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

    def calculate(self, coordinates: jnp.array):
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
        neighbor_list: jnp.array
            Array of particle ids for the neighbors, padded to n_max_neighbors. Shape (n_particles, n_max_neighbors)
        padding_mask: jnp.array
            Array of masks to exclude padding from the neighbor list of each particle. Shape (n_particles, n_max_neighbors)
        dist: jnp.array
            Array of distances between each particle and its neighbors. Shape (n_particles, n_max_neighbors)
        r_ij: jnp.array
            Array of displacement vectors between each particle and its neighbors. Shape (n_particles, n_max_neighbors, 3)
        """
        # coordinates = sampler_state.x0
        # note, we assume the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list

        n_neighbors, padding_mask, dist, r_ij = jax.vmap(
            self._calc_distance_per_particle, in_axes=(0, 0, 0, None)
        )(self.particle_ids, self.neighbor_list, self.neighbor_mask, coordinates)
        # mask = mask.reshape(-1, self.n_max_neighbors)
        return n_neighbors, self.neighbor_list, padding_mask, dist, r_ij

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

    def check(self, coordinates: jnp.array) -> bool:
        """
        Check if the neighbor list needs to be rebuilt based on displacement of the particles from the reference coordinates.
        If a particle moves more than 0.5 skin distance, the neighborlist will be rebuilt.
        Will also return True if the size of the coordinates array changes.

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

        if self.ref_coordinates.shape[0] != coordinates.shape[0]:
            return True

        status = jax.vmap(
            self._calculate_particle_displacement, in_axes=(0, None, None)
        )(self.particle_ids, coordinates, self.ref_coordinates)
        if jnp.any(status):
            del status
            return True
        else:
            del status
            return False


class PairList(PairsBase):
    """
    N^2 pairlist implementation that returns the particle pair ids, displacement vectors, and distances.

    Parameters
    ----------
    space: Space
        Class that defines how to calculate the displacement between two points and apply the boundary conditions
    cutoff: float, default = 2.5
        Cutoff distance for the pair list calculation
    Examples
    --------
    >>> from chiron.neighbors import PairList, OrthogonalPeriodicSpace
    >>> from chiron.states import SamplerState
    >>> import jax.numpy as jnp
    >>>
    >>> space = OrthogonalPeriodicSpace()
    >>> pair_list = PairList(space, cutoff=2.5)
    >>> sampler_state = SamplerState(x0=jnp.array([[0.0, 0.0, 0.0], [2, 0.0, 0.0], [0.0, 2, 0.0]]),
    >>>                                 box_vectors=jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]]))
    >>> pair_list.build_from_state(sampler_state)
    >>>
    >>> # mask and distances are of shape (n_particles, n_particles-1),
    >>> displacement_vectors of shape (n_particles, n_particles-1, 3)
    >>> # mask, is a bool array that is True if the particle is within the cutoff distance, False if it is not
    >>> # n_pairs is of shape (n_particles) and is per row sum of the mask. The mask ensure we also do not double count pairs
    >>> n_pairs, mask, distances, displacement_vectors = pair_list.calculate(sampler_state.x0)
    """

    def __init__(
        self,
        space: Space,
        cutoff: unit.Quantity = unit.Quantity(1.2, unit.nanometer),
    ):
        if not isinstance(space, Space):
            raise TypeError(f"space must be of type Space, found {type(space)}")
        if not cutoff.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, cutoff.unit = {cutoff.unit}"
            )

        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)
        self.space = space

        # set a a simple variable to know if this has at least been built once as opposed to just initialized
        # this does not imply that the neighborlist is up to date
        self.is_built = False

    # note, we need to use the partial decorator in order to use the jit decorate
    # so that it knows to ignore the `self` argument
    @partial(jax.jit, static_argnums=(0,))
    def _pairs_and_mask(self, particle_ids: jnp.array):
        """
        Jitted function to generate all pairs (excluding self interactions)
        and  mask that allows us to remove double-counting of pairs.

        Parameters
        ----------
        particle_ids: jnp.array
            Array of particle ids

        Returns
        -------
        all_pairs: jnp.array
            Array of all pairs (excluding self interactions), of size (n_particles, n_particles-1)
        reduction_mask: jnp.array
            Bool mask that identifies which pairs to exclude to remove double counting of pairs

        """
        # for the nsq approach, we consider the distance between a particle and all other particles in the system
        # if we used a cell list the possible_neighbors would be a smaller list, i.e., only those in the neigboring cells
        # we'll just keep with naming syntax for future flexibility

        possible_neighbors = particle_ids

        particles_j = jnp.broadcast_to(
            possible_neighbors,
            (particle_ids.shape[0], possible_neighbors.shape[0]),
        )
        # reshape the particle_ids
        particles_i = jnp.reshape(particle_ids, (particle_ids.shape[0], 1))
        # create a mask to exclude self interactions and double counting
        temp_mask = particles_i != particles_j
        all_pairs = jax.vmap(self._remove_self_interactions, in_axes=(0, 0))(
            particles_j, temp_mask
        )
        del temp_mask
        all_pairs = jnp.array(all_pairs[0], dtype=jnp.uint32)

        reduction_mask = jnp.where(particles_i < all_pairs, True, False)

        return all_pairs, reduction_mask

    @partial(jax.jit, static_argnums=(0,))
    def _remove_self_interactions(self, particles, temp_mask):
        return jnp.where(
            temp_mask, size=particles.shape[0] - 1, fill_value=particles.shape[0] - 1
        )

    def build(
        self,
        coordinates: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build the neighborlist from an array of coordinates and box vectors.

        Parameters
        ----------
        coordinates: jnp.array
            Shape[n_particles,3] array of particle coordinates
        box_vectors: jnp.array
            Shape[3,3] array of box vectors

        Returns
        -------
        None

        """

        # set our reference coordinates
        # this will set self.ref_coordinates=coordinates and self.box_vectors
        self._validate_build_inputs(coordinates, box_vectors)

        self.n_particles = self.ref_coordinates.shape[0]

        # the neighborlist assumes that the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list
        self.space.box_vectors = self.box_vectors

        # store the ids of all the particles
        self.particle_ids = jnp.array(range(0, coordinates.shape[0]), dtype=jnp.uint32)

        # calculate which pairs to exclude
        self.all_pairs, self.reduction_mask = self._pairs_and_mask(self.particle_ids)

        self.is_built = True

    @partial(jax.jit, static_argnums=(0,))
    def _calc_distance_per_particle(
        self, particle1, neighbors, neighbor_mask, coordinates
    ):
        """
        Jitted function to calculate the distance between a particle and all possible neighbors

        Parameters
        ----------
        particle1: int
            Particle id
        neighbors: jnp.array
            Array of particle ids for the possible particle pairs of particle1
        neighbor_mask: jnp.array
            Mask to exclude double particles to prevent double counting
        coordinates: jnp.array
            X,Y,Z coordinates of all particles, shaped (n_particles, 3)

        Returns
        -------
        n_pairs: int
            Number of interacting pairs for the particle
        mask: jnp.array
            Mask to exclude padding particles not within the cutoff particle1.
            If a particle is within the interaction cutoff, the mask is 1, otherwise it is 0
            Array has shape (n_particles, n_particles-1) as it excludes self interactions
        dist: jnp.array
            Array of distances between the particle and all other particles in the system.
            Array has shape (n_particles, n_particles-1) as it excludes self interactions
        r_ij: jnp.array
            Array of displacement vectors between the particle and all other particles in the system.
            Array has shape (n_particles, n_particles-1, 3) as it excludes self interactions

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

    def calculate(self, coordinates: jnp.array):
        """
        Calculate the neighbor list for the current state

        Parameters
        ----------
        coordinates: jnp.array
            Shape[n_particles,3] array of particle coordinates

        Returns
        -------
        n_neighbors: jnp.array
            Array of the number of interacting particles (i.e., where dist < cutoff). Shape: (n_particles)
        pairs: jnp.array
            Array of particle ids that were considered for interaction. Shape: (n_particles, n_particles-1)
        padding_mask: jnp.array
            Array used to masks non interaction particle pairs. Shape: (n_particles, n_particles-1)
        dist: jnp.array
            Array of distances between pairs in the system. Shape: (n_particles, n_particles-1)
        r_ij: jnp.array
            Array of displacement vectors between particle pairs. Shape: (n_particles, n_particles-1, 3).
        """
        if coordinates.shape[0] != self.n_particles:
            raise ValueError(
                f"Number of particles cannot changes without rebuilding. "
                f"Coordinates must have shape ({self.n_particles}, 3), found {coordinates.shape}"
            )

        n_neighbors, padding_mask, dist, r_ij = jax.vmap(
            self._calc_distance_per_particle, in_axes=(0, 0, 0, None)
        )(self.particle_ids, self.all_pairs, self.reduction_mask, coordinates)

        return n_neighbors, self.all_pairs, padding_mask, dist, r_ij

    def check(self, coordinates: jnp.array) -> bool:
        """
        Check if we need to reconstruct internal arrays.
        For a simple pairlist this will always return False, unless the number of particles change.

        Parameters
        ----------
        coordinates: jnp.array
            Array of particle coordinates
        Returns
        -------
        bool
            True if we need to rebuild the neighbor list, False if we do not.
        """
        if coordinates.shape[0] != self.n_particles:
            return True
        else:
            return False
