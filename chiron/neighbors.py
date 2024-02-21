# This file contains various routines related to generating pair lists

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Union, Optional
from .states import SamplerState
from openmm import unit


# split out the displacement calculation from the neighbor list and pair list for flexibility
from abc import ABC, abstractmethod


class Space(ABC):
    """
    Abstract Base Class for different simulation spaces.

    This class will define two functions:
     - displacement, i.e., how to calculate the displacement vector and distance between two points
     - wrap, i.e., how to wrap a particle in the box (i.e., apply boundary conditions).

    Note, this class does not store the box_vectors; they will need to be passed to each function.


    """

    @abstractmethod
    def displacement(
        self, xyz_1: jnp.array, xyz_2: jnp.array, box_vectors: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        pass

    @abstractmethod
    def wrap(self, xyz: jnp.array, box_vectors: jnp.array) -> jnp.array:
        pass


class OrthogonalPeriodicSpace(Space):
    """
    Defines the simulation space for an orthogonal periodic system.

    """

    @partial(jax.jit, static_argnums=(0,))
    def displacement(
        self, xyz_1: jnp.array, xyz_2: jnp.array, box_vectors: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Positions of the first point
        xyz_2: jnp.array
            Positions of the second point
        box_vectors: jnp.array

        Returns
        -------
        r_ij: jnp.array
            Displacement vector between the two points
        dist: float
            Distance between the two points

        """
        # calculate uncorrected r_ij
        r_ij = xyz_1 - xyz_2

        box_lengths = jnp.array(
            [box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]]
        )
        # calculated corrected displacement vector
        # using modulus seems faster in JAX
        r_ij = jnp.mod(r_ij + box_lengths * 0.5, box_lengths) - box_lengths * 0.5
        # calculate the scalar distance
        dist = jnp.linalg.norm(r_ij, axis=-1)

        return r_ij, dist

    @partial(jax.jit, static_argnums=(0,))
    def wrap(self, xyz: jnp.array, box_vectors: jnp.array) -> jnp.array:
        """
        Wrap the positions of the system.

        Parameters
        ----------
        xyz: jnp.array
            Positions of the system
        box_vectors: jnp.array
            Box vectors for the system

        Returns
        -------
        jnp.array
            Wrapped positions of the system

        """
        box_lengths = jnp.array(
            [box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]]
        )

        xyz = xyz - jnp.floor(xyz / box_lengths) * box_lengths

        return xyz


class OrthogonalNonperiodicSpace(Space):
    @partial(jax.jit, static_argnums=(0,))
    def displacement(
        self,
        xyz_1: jnp.array,
        xyz_2: jnp.array,
        box_vectors: jnp.array,
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate the periodic distance between two points.

        Parameters
        ----------
        xyz_1: jnp.array
            Positions of the first point
        xyz_2: jnp.array
            Positions of the second point
        box_vectors: jnp.array
            Box vectors for the system.

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
    def wrap(self, xyz: jnp.array, box_vectors: jnp.array) -> jnp.array:
        """
        Wrap the positions of the system.
        For the Non-periodic system, this does not alter the positions

        Parameters
        ----------
        xyz: jnp.array
            Positions of the system
        box_vectors: jnp.array
            Box vectors for the system

        Returns
        -------
        jnp.array
            Wrapped positions of the system

        """
        return xyz


class PairsBase(ABC):
    """
    Abstract Base Class for different algorithms that determine which particle pairs are interacting.

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
    >>> sampler_state = SamplerState(positions=jnp.array([[0.0, 0.0, 0.0], [2, 0.0, 0.0], [0.0, 2, 0.0]]),
    >>>                              box_vectors=jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]]))
    >>>
    >>> pair_list = PairsBase(space, cutoff=2.5*unit.nanometer) # initialize the pair list
    >>> pair_list.build_from_state(sampler_state) # build the pair list from the sampler state
    >>>
    >>> positions = sampler_state.positions # get the positions from the sampler state, without units attached
    >>>
    >>> # the calculate function will produce information used to calculate the energy
    >>> n_neighbors, padding_mask, dist, r_ij = pair_list.calculate(positions)
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
        self.cutoff = cutoff
        self.space = space

    @abstractmethod
    def build(
        self,
        positions: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build list from an array of positions and array of box vectors.

        Parameters
        ----------
        positions: jnp.array or unit.Quantity
            Shape[n_particles,3] array of particle positions, either with or without units attached.
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
        positions: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Validate the inputs to the build function.

        This will raise ValueErrors if the inputs are not of the correct type or shape or compatible units

        Parameters
        ----------
        positions: jnp.array or unit.Quantity
            Shape[n_particles,3] array of particle positions, either with or without units attached.
            If the array is passed as a unit.Quantity, the units must be distances and will be converted to nanometers.
        box_vectors: jnp.array or unit.Quantity
            Shape[3,3] array of box vectors for the system, either with or without units attached.
            If the array is passed as a unit.Quantity, the units must be distances and will be converted to nanometers.

        """
        if isinstance(positions, unit.Quantity):
            if not positions.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Positions require distance units, not {positions.unit}"
                )
            self.ref_positions = positions.value_in_unit_system(unit.md_unit_system)
        if isinstance(positions, jnp.ndarray):
            if positions.shape[1] != 3:
                raise ValueError(
                    f"positions should be a Nx3 array, shape provided: {positions.shape}"
                )
            self.ref_positions = positions
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
            SamplerState object containing the positions and box vectors

        Returns
        -------
        None
        """
        if not isinstance(sampler_state, SamplerState):
            raise TypeError(f"Expected SamplerState, got {type(sampler_state)} instead")

        positions = sampler_state.positions
        if sampler_state.box_vectors is None:
            raise ValueError(f"SamplerState does not contain box vectors")
        box_vectors = sampler_state.box_vectors

        self.build(positions, box_vectors)

    @abstractmethod
    def calculate(self, positions: jnp.array):
        """
        Calculate the list of interacting particles for the current state

        Parameters
        ----------
        positions: jnp.array
            Shape[N,3] array of particle positions

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
    def check(self, positions: jnp.array) -> bool:
        """
        Check if the internal variables need to be reset. E.g., rebuilding a neighborlist if particles moved to far,
        or rebuilding if number of particles changes.


        Parameters
        ----------
        positions: jnp.array
            Array of particle positions
        Returns
        -------
        bool
            True if the neighbor list needs to be rebuilt, False if it does not.
        """
        pass


class NeighborListNsqrd(PairsBase):
    """
        Neighbor list implementation used to determine which particles are interacting (i.e., within the cutoff).

        This `calculate` function of this class returns the particle pair ids, displacement vectors, and distances
        between pairs within the specified cutoff range, subject to the boundary conditions defined by the simulation
        Space class passed to the constructor.

        The neighbor list (i.e., list of particles within cutoff+skin of a given particle)
        is generated using an N^2 calculation rather than using a spatial partitioning scheme (e.g., cell-list).


        Notes:
        This does not include self-interactions and only includes unique pairs (i.e., no double-counting).
        This is sometimes referred to as a "half" neighbor list. E.g. consider the pair of neighboring particles (A, B):
        in the "half" neighbor list approach, B is in the neighbor list of A, but A is not in the neighbor list of B
        as that pair is already accounted for.
    .
        The output of the `calculate` function is padded to a fixed size, n_max_neighbors, to allow for efficient
        jitted computations in JAX. As such, values need to be masked using the `padding_mask` array returned by the
        `calculate` function.  The padding mask is simple an array of 1s and 0s, where 1 indicates a valid neighbor and
        0 indicates padding. The code will automatically increase n_max_neighbors by 10 if the number of neighbors
        exceeds the current value.

        The check function, which indicates if the neighbor list should be rebuilt, will return True if:
        - the number of particles changes
        - any of the particles have moved more than half the skin distance from their positions at the time of last
        building the neighbor list.


        Parameters
        ----------
        space: Space
            Class that defines how to calculate the displacement between two points and apply the boundary conditions
        cutoff: unit.Quantity, default = 1.2 unit.nanometer
            Cutoff distance for the neighborlist
        skin: unit.Quantity, default = 0.4 unit.nanometer
            Skin distance, i.e., buffer, for the neighborlist
            Larger values of the skin will reduce the frequency of rebuilding the neighbor list,
            but will increase the number of neighbors to consider.
        n_max_neighbors: int, default=200
            Maximum number of neighbors for each particle.  This is used for padding arrays for efficient jax computations
            n_max_neighbors will be dynamically updated (in increments of 10) as part of the build function.
        Examples
        --------
        >>> from openmm import unit
        >>> import jax.numpy as jnp
        >>>
        >>> from chiron.states import SamplerState
        >>> sampler_state = SamplerState(positions=jnp.array([[0.0, 0.0, 0.0], [2, 0.0, 0.0], [0.0, 2, 0.0]])*unit.nanometer,
        >>>                              box_vectors=jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]])*unit.nanometer)
        >>>
        >>> from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace
        >>> nbr_list = NeighborListNsqrd(OrthogonalPeriodicSpace(), cutoff=1.2*unit.nanometer, skin=0.4*unit.nanometer)
        >>>
        >>> # build the neighborlist
        >>> nbr_list.build_from_state(sampler_state) # build the pair list from the sampler state
        >>>
        >>> # calculate which particles are interacting along with their distances and displacement vectors
        >>> n_neighbors, neighbor_list, padding_mask, dist, r_ij = nbr_list.calculate(sampler_state.positions)
        >>>
        >>> # check the neighborlist
        >>> if nbr_list.check(sampler_state.positions):
        >>>     nbr_list.build_from_state(sampler_state) # rebuild the pair list from the sampler state

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

        if not skin.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, skin.unit = {skin.unit}"
            )

        self.cutoff = cutoff
        self.skin = skin
        self.n_max_neighbors = n_max_neighbors
        self.space = space

        # set a a simple variable to know if this has at least been built once as opposed to just initialized
        # this does not imply that the neighborlist is up to date
        self.is_built = False

    @property
    def cutoff(self) -> unit.Quantity:
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff: unit.Quantity) -> None:
        if not cutoff.unit.is_compatible(unit.nanometer):
            raise ValueError(
                f"cutoff must be a unit.Quantity with units of distance, cutoff.unit = {cutoff.unit}"
            )
        self._cutoff = cutoff

        # if we change the cutoff or skin we need to rebuild
        self.is_built = False

    @property
    def skin(self) -> unit.Quantity:
        return self._skin

    @skin.setter
    def skin(self, skin: unit.Quantity) -> None:
        if not skin.unit.is_compatible(unit.nanometer):
            raise ValueError(
                f"skin must be a unit.Quantity with units of distance, skin.unit = {skin.unit}"
            )
        self._skin = skin

        # if we change the cutoff or skin we need to rebuild
        self.is_built = False

    # Note, we need to use the partial decorator in order to jit the method of a class
    # so that it knows to ignore the `self` argument.  However, this treats it as static.
    # This means that any changes to the internal class values within the class will not be updated in this function.
    # Hence it is important that we pass the values as arguments to the function so we are getting updated values
    # and not reference anything via self.variable_name.
    # Alternatively we could create a custom pytree instead of declaring the class static in this function,
    # but I don't think that is necessary if we just pass the values as arguments.

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

    # note: since n_max_neighbors dictates the output size, we need to define it as a static argument
    # to allow us to jit this function
    @partial(jax.jit, static_argnums=(0, 5))
    def _build_neighborlist(
        self,
        particle_i,
        reduction_mask,
        pid,
        positions,
        n_max_neighbors,
        cutoff_and_skin,
        box_vectors,
    ):
        """
        Jitted function to build the neighbor list for a single particle

        Parameters
        ----------
        particle_i: jnp.array
            X,Y,Z positions of particle i
        reduction_mask: jnp.array
            Mask to exclude self-interactions and double counting of pairs
        positions: jnp.array
            X,Y,Z positions of all particles
        n_max_neighbors: int
            Maximum number of neighbors for each particle.  Used for padding arrays for efficient jax computations
        cutoff_and_skin: float
            Cutoff distance for the neighborlist plus the skin distance, in nanometers.
        box_vectors: jnp.array
            Box vectors for the system

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
        # we could pass this as a function instead of referencing self.space, but I ran into issues
        # doing that with vmap, and I haven't been able to figure out how to resolve that yet  -- CRI
        r_ij, dist = self.space.displacement(particle_i, positions, box_vectors)

        # neighbor_mask will be an array of length n_particles (i.e., length of positions)
        # where each element is True if the particle is a neighbor, False if it is not
        # subject to both the cutoff+skin and the reduction mask that eliminates double counting and self-interactions
        neighbor_mask = jnp.where(
            (dist < cutoff_and_skin) & (reduction_mask), True, False
        )
        # when we  pad the neighbor list, we will use last particle id in the neighbor list
        # this choice was made such that when we use the neighbor list in the masked energy calculation
        # the padded values will result in reasonably well defined values
        fill_value = jnp.argmax(neighbor_mask)
        # if the max value is the same as the particle of interest, which can occur if particle 0 has no neighbors
        # we will just increment by 1 to avoid calculating a self interaction
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
        positions: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build the neighbor list from an array of positions and box vectors.

        Parameters
        ----------
        positions: jnp.array
            Shape[N,3] array of particle positions
        box_vectors: jnp.array
            Shape[3,3] array of box vectors

        Returns
        -------
        None

        """

        # set our reference positions
        # the call to positions and box_vectors automatically convert these to jnp arrays in the correct unit system
        if isinstance(positions, unit.Quantity):
            if not positions.unit.is_compatible(unit.nanometer):
                raise ValueError(
                    f"Positions require distance units, not {positions.unit}"
                )
            positions = positions.value_in_unit_system(unit.md_unit_system)

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

        self.ref_positions = positions
        self.box_vectors = box_vectors

        cutoff_and_skin = self.cutoff + self.skin

        # the neighborlist assumes that the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list

        # store the ids of all the particles
        self.particle_ids = jnp.array(
            range(0, self.ref_positions.shape[0]), dtype=jnp.uint32
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
            self._build_neighborlist, in_axes=(0, 0, 0, None, None, None, None)
        )(
            self.ref_positions,
            reduction_mask,
            self.particle_ids,
            self.ref_positions,
            self.n_max_neighbors,
            cutoff_and_skin.value_in_unit_system(unit.md_unit_system),
            self.box_vectors,
        )

        self.neighbor_list = self.neighbor_list.reshape(-1, self.n_max_neighbors)
        from loguru import logger as log

        while jnp.any(self.n_neighbors == self.n_max_neighbors).block_until_ready():
            log.debug(
                f"Increasing n_max_neighbors from {self.n_max_neighbors} to at  {jnp.max(self.n_neighbors)+10}"
            )
            self.n_max_neighbors = int(jnp.max(self.n_neighbors) + 10)

            self.neighbor_mask, self.neighbor_list, self.n_neighbors = jax.vmap(
                self._build_neighborlist, in_axes=(0, 0, 0, None, None, None, None)
            )(
                self.ref_positions,
                reduction_mask,
                self.particle_ids,
                self.ref_positions,
                self.n_max_neighbors,
                cutoff_and_skin.value_in_unit_system(unit.md_unit_system),
                self.box_vectors,
            )

            self.neighbor_list = self.neighbor_list.reshape(-1, self.n_max_neighbors)

        self.is_built = True

    @partial(jax.jit, static_argnums=(0,))
    def _calc_distance_per_particle(
        self, particle1, neighbors, neighbor_mask, positions, cutoff, box_vectors
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
        positions: jnp.array
            X,Y,Z positions of all particles
        cutoff: float
            Cutoff distance for the neighborlist, in nanometers
        box_vectors: jnp.array
            Box vectors for the system

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
            positions[particles1], positions[neighbors], box_vectors
        )
        # calculate the mask to determine if the particle is a neighbor
        # this will be done based on the interaction cutoff and using the neighbor_mask to exclude padding
        mask = jnp.where((dist < cutoff) & (neighbor_mask), 1, 0)

        # calculate the number of pairs
        n_pairs = mask.sum()

        return n_pairs, mask, dist, r_ij

    def calculate(self, positions: jnp.array):
        """
        Calculate the neighbor list for the current state

        Parameters
        ----------
        positions: jnp.array
            Shape[N,3] array of particle positions

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
        # positions = sampler_state.positions
        # note, we assume the box vectors do not change between building and calculating the neighbor list
        # changes to the box vectors require rebuilding the neighbor list

        n_neighbors, padding_mask, dist, r_ij = jax.vmap(
            self._calc_distance_per_particle, in_axes=(0, 0, 0, None, None, None)
        )(
            self.particle_ids,
            self.neighbor_list,
            self.neighbor_mask,
            positions,
            self.cutoff.value_in_unit_system(unit.md_unit_system),
            self.box_vectors,
        )
        # mask = mask.reshape(-1, self.n_max_neighbors)
        return n_neighbors, self.neighbor_list, padding_mask, dist, r_ij

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_particle_displacement(
        self,
        particle: int,
        positions: jnp.array,
        ref_positions: jnp.array,
        skin: float,
        box_vectors: jnp.array,
    ):
        """
        Calculate the displacement of a particle from the reference positions.
        If the displacement exceeds the half the skin distance, return True, otherwise return False.

        This function is designed to allow it to be jitted and vmapped over particle indices.

        Parameters
        ----------
        particle: int
            Particle id
        positions: jnp.array
            Array of particle positions
        ref_positions: jnp.array
            Array of reference particle positions
        skin: float
            Skin distance for the neighborlist, in nanometers
        box_vectors: jnp.array
            Box vectors for the system


        Returns
        -------
        bool
            True if the particle is outside the skin distance, False if it is not.
        """
        # calculate the displacement of a particle from the initial positions

        r_ij, displacement = self.space.displacement(
            positions[particle], ref_positions[particle], box_vectors
        )

        status = jnp.where(displacement >= skin / 2.0, True, False)
        del displacement
        return status

    def check(self, positions: jnp.array) -> bool:
        """
        Check if the neighbor list needs to be rebuilt based on displacement of the particles from the reference positions.
        If a particle moves more than 0.5 skin distance, the neighborlist will be rebuilt.
        Will also return True if the size of the positions array changes.

        Note, this could also accept a user defined criteria for distance, but this is not implemented yet.

        Parameters
        ----------
        positions: jnp.array
            Array of particle positions
        Returns
        -------
        bool
            True if the neighbor list needs to be rebuilt, False if it does not.
        """

        if self.ref_positions.shape[0] != positions.shape[0]:
            return True

        status = jax.vmap(
            self._calculate_particle_displacement, in_axes=(0, None, None, None, None)
        )(
            self.particle_ids,
            positions,
            self.ref_positions,
            self.skin.value_in_unit_system(unit.md_unit_system),
            self.box_vectors,
        )
        if jnp.any(status):
            del status
            return True
        else:
            del status
            return False


class PairListNsqrd(PairsBase):
    """
    Pair list implementation to determine which particles are interacting, subject to the simulation
    Space class. This performs an N^2 calculation to determine the distances between particles which will be
    inefficient for all but very small system sizes.  This routine can be defined either with or without a cutoff.

    The `calculate` function of this class returns the particle pair ids, displacement vectors, and distances.
    For efficiency of the jitted functions, the `calculate` function array sizes are fixed. For example, distance
    has shape (n_particles, n_particles-1); note self-interactions are removed, hence n_particles-1).
    The masking vector contains values of 1 for interacting particles and 0 otherwise.

    Parameters
    ----------
    space: Space
        Class that defines how to calculate the displacement between two points and apply the boundary conditions
    cutoff: Optional[unit.Quantity], default = None
        Cutoff distance for the pair list calculation.  If None, the pair list will be calculated without a cutoff,
        applying the boundary conditions as defined in space.
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import openmm.unit as unit
    >>>
    >>> from chiron.states import SamplerState
    >>> sampler_state = SamplerState(positions=jnp.array([[0.0, 0.0, 0.0], [2, 0.0, 0.0], [0.0, 2, 0.0]]),
    >>>                                 box_vectors=jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]]))
    >>>
    >>> from chiron.neighbors import PairListNsqrd, OrthogonalPeriodicSpace
    >>> pair_list = PairListNsqrd(OrthogonalPeriodicSpace(), cutoff=1.2*unit.nanometer)
    >>> pair_list.build_from_state(sampler_state)
    >>>
    >>> # mask and distances are of shape (n_particles, n_particles-1),
    >>> # displacement_vectors of shape (n_particles, n_particles-1, 3)
    >>> # mask, is a bool array that is True if the particle is within the cutoff distance, False if it is not
    >>> # n_pairs is of shape (n_particles) and is per row sum of the mask. The mask ensure we also do not double count pairs
    >>> n_pairs, mask, distances, displacement_vectors = pair_list.calculate(sampler_state.positions)
    """

    def __init__(
        self,
        space: Space,
        cutoff: Optional[unit.Quantity] = None,
    ):
        if not isinstance(space, Space):
            raise TypeError(f"space must be of type Space, found {type(space)}")

        # keeping this public in case we want to change it later
        # validation is performed in the setter
        self.cutoff = cutoff

        self.space = space

        # set a a simple variable to know if this has at least been built once as opposed to just initialized
        # this does not imply that the neighborlist is up to date
        self.is_built = False

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff):
        if cutoff is not None:
            if not cutoff.unit.is_compatible(unit.angstrom):
                raise ValueError(
                    f"cutoff must be a unit.Quantity with units of distance, cutoff.unit = {cutoff.unit}"
                )
        self._cutoff = cutoff
        # Since this is just a simple pair list, we do not need to rebuild by changing a cutoff

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
        # remove self interactions
        all_pairs = jax.vmap(self._remove_self_interactions, in_axes=(0, 0))(
            particles_j, temp_mask
        )
        del temp_mask
        all_pairs = jnp.array(all_pairs[0], dtype=jnp.uint32)

        # create the mask that will remove any double counting of pairs
        reduction_mask = jnp.where(particles_i < all_pairs, True, False)

        return all_pairs, reduction_mask

    @partial(jax.jit, static_argnums=(0,))
    def _remove_self_interactions(self, particles, temp_mask):
        return jnp.where(
            temp_mask, size=particles.shape[0] - 1, fill_value=particles.shape[0] - 1
        )

    def build(
        self,
        positions: Union[jnp.array, unit.Quantity],
        box_vectors: Union[jnp.array, unit.Quantity],
    ):
        """
        Build the list from an array of positions and box vectors.

        Parameters
        ----------
        positions: jnp.array
            Shape[n_particles,3] array of particle positions
        box_vectors: jnp.array
            Shape[3,3] array of box vectors

        Returns
        -------
        None

        """

        # validate the positions and box vectors
        self._validate_build_inputs(positions, box_vectors)

        self.n_particles = self.ref_positions.shape[0]

        # the PairsList assumes that the box vectors do not change between building and calculating the neighbor list

        # store the ids of all the particles
        self.particle_ids = jnp.array(range(0, positions.shape[0]), dtype=jnp.uint32)

        # calculate which pairs to exclude
        self.all_pairs, self.reduction_mask = self._pairs_and_mask(self.particle_ids)

        self.is_built = True

    @partial(jax.jit, static_argnums=(0))
    def _calc_distance_per_particle_with_cutoff(
        self, particle1, neighbors, neighbor_mask, positions, cutoff, box_vectors
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
                positions: jnp.array
                    X,Y,Z positions of all particles, shaped (n_particles, 3)
                cutoff: float
                    Cutoff distance for the interaction.
                box_vectors: jnp.array
                    Box vectors for the system

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
        .

        """
        # repeat the particle id for each neighbor
        particles1 = jnp.repeat(particle1, neighbors.shape[0])

        # calculate the displacement between particle i and all  neighbors
        r_ij, dist = self.space.displacement(
            positions[particles1], positions[neighbors], box_vectors
        )
        # calculate the mask to determine if the particle is a neighbor
        # this will be done based on the interaction cutoff and using the neighbor_mask to exclude padding
        mask = jnp.where((dist < cutoff) & (neighbor_mask), 1, 0)

        # calculate the number of pairs
        n_pairs = mask.sum()

        return n_pairs, mask, dist, r_ij

    @partial(jax.jit, static_argnums=(0))
    def _calc_distance_per_particle_no_cutoff(
        self, particle1, neighbors, neighbor_mask, positions, box_vectors
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
        positions: jnp.array
            X,Y,Z positions of all particles, shaped (n_particles, 3)
        box_vectors: jnp.array
            Box vectors of the system

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
            positions[particles1], positions[neighbors], box_vectors
        )
        # calculate the mask to determine if the particle is a neighbor
        # this will be done based on the interaction cutoff and using the neighbor_mask to exclude padding
        mask = jnp.where(neighbor_mask, 1, 0)

        # calculate the number of pairs
        n_pairs = mask.sum()

        return n_pairs, mask, dist, r_ij

    def calculate(self, positions: jnp.array):
        """
        Calculate the list of neighbor pairs for the current state

        Parameters
        ----------
        positions: jnp.array
            Shape[n_particles,3] array of particle positions

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
        if positions.shape[0] != self.n_particles:
            raise ValueError(
                f"Number of particles cannot changes without rebuilding. "
                f"Positions must have shape ({self.n_particles}, 3), found {positions.shape}"
            )

        # if we did not define a cutoff, we will
        if self.cutoff is None:
            n_neighbors, padding_mask, dist, r_ij = jax.vmap(
                self._calc_distance_per_particle_no_cutoff,
                in_axes=(0, 0, 0, None, None),
            )(
                self.particle_ids,
                self.all_pairs,
                self.reduction_mask,
                positions,
                self.box_vectors,
            )
        else:
            n_neighbors, padding_mask, dist, r_ij = jax.vmap(
                self._calc_distance_per_particle_with_cutoff,
                in_axes=(0, 0, 0, None, None, None),
            )(
                self.particle_ids,
                self.all_pairs,
                self.reduction_mask,
                positions,
                self.cutoff.value_in_unit_system(unit.md_unit_system),
                self.box_vectors,
            )
        return n_neighbors, self.all_pairs, padding_mask, dist, r_ij

    def check(self, positions: jnp.array) -> bool:
        """
        Check if we need to reconstruct internal arrays.
        For a simple pairlist this will always return False, unless the number of particles change.

        Parameters
        ----------
        positions: jnp.array
            Array of particle positions
        Returns
        -------
        bool
            True if we need to rebuild the neighbor list, False if we do not.
        """
        if positions.shape[0] != self.n_particles:
            return True
        else:
            return False
