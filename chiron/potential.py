import jax
import jax.numpy as jnp
from loguru import logger as log
from openmm import unit
from openmm.app import Topology
from typing import Optional


class NeuralNetworkPotential:
    def __init__(self, model, **kwargs):
        if model is None:
            log.warning("No model provided, using default model")
        else:
            self.model = model  # The trained neural network model
            self.topology = model.potential.topology  # The topology of the system

    def compute_energy(self, positions, nbr_list=None):
        # Compute the pair distances and displacement vectors
        raise NotImplementedError

    def compute_force(self, positions, nbr_list=None) -> jnp.ndarray:
        # Compute the force as the negative gradient of the potential energy
        force = -jax.grad(self.compute_energy)(positions, nbr_list)
        return force

    def compute_pairlist(self, positions, cutoff) -> jnp.array:
        """
        Compute the pairlist for a given set of positions and a cutoff distance
        without using periodic boundary conditions or any specific optimizations.

        Parameters
        ----------
        positions : jnp.array
            The positions of the particles in the system
        cutoff : float
            The cutoff distance for the pairlist

        Returns
        -------
        distance : jnp.array
            The distances between all pairs of particles
        displacement_vectors : jnp.array
            The displacement vectors between all pairs of particles
        pairs : jnp.array
            An array of the particle pairs shaped (2, n_pairs)

        """

        pids = jnp.arange(positions.shape[0])
        pairs1, pairs2 = jnp.meshgrid(pids, pids)
        pairs1 = pairs1.flatten()
        pairs2 = pairs2.flatten()
        mask = jnp.where(pairs1 < pairs2)

        pairs1 = pairs1[mask]
        pairs2 = pairs2[mask]
        displacement_vectors = positions[pairs1] - positions[pairs2]
        distance = jnp.linalg.norm(displacement_vectors, axis=1)
        interacting_mask = jnp.where(distance < cutoff)

        # exclude case where i ==j and duplicate pairs
        pairs = jnp.stack((pairs1[interacting_mask], pairs2[interacting_mask]), axis=0)
        return distance[interacting_mask], displacement_vectors[interacting_mask], pairs


class LJPotential(NeuralNetworkPotential):
    def __init__(
        self,
        topology: Topology,
        sigma: unit.Quantity = 3.350 * unit.angstroms,
        epsilon: unit.Quantity = 1.0 * unit.kilocalories_per_mole,
        cutoff: unit.Quantity = unit.Quantity(1.0, unit.nanometer),
    ):
        """
        Initialize the Lennard-Jones potential.

        Parameters
        ----------
        topology : Topology
            The topology of the system
        sigma : unit.Quantity, optional
            The distance at which the potential is zero, by default 3.350 * unit.angstroms
        epsilon : unit.Quantity, optional
            The depth of the potential well, by default 1.0 * unit.kilocalories_per_mole
        cutoff : unit.Quantity, optional
            The cutoff distance for the potential, by default 1.0 * unit.nanometer

        """

        if not isinstance(topology, Topology):
            if not isinstance(topology, property):
                if topology is not None:
                    raise TypeError(
                        f"Topology must be a Topology object or None, type(topology) = {type(topology)}"
                    )
        if not isinstance(sigma, unit.Quantity):
            raise TypeError(
                f"sigma must be a unit.Quantity, type(sigma) = {type(sigma)}"
            )
        if not isinstance(epsilon, unit.Quantity):
            raise TypeError(
                f"epsilon must be a unit.Quantity, type(epsilon) = {type(epsilon)}"
            )
        if not isinstance(cutoff, unit.Quantity):
            raise TypeError(
                f"cutoff must be a unit.Quantity, type(cutoff) = {type(cutoff)}"
            )

        if not sigma.unit.is_compatible(unit.angstrom):
            raise ValueError(f"sigma must have units of distance, got {sigma.unit}")
        if not epsilon.unit.is_compatible(unit.kilocalories_per_mole):
            raise ValueError(f"epsilon must have units of energy, got  {epsilon.unit}")
        if not cutoff.unit.is_compatible(unit.nanometer):
            raise ValueError(f"cutoff must have units of distance, got {cutoff.unit}")

        self.sigma = sigma.value_in_unit_system(
            unit.md_unit_system
        )  # The distance at which the potential is zero
        self.epsilon = epsilon.value_in_unit_system(
            unit.md_unit_system
        )  # The depth of the potential well
        # The cutoff for a potential is often linked with the parameters and isn't really
        # something I think we should be changing dynamically.
        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)
        self.topology = topology

    from functools import partial

    @partial(jax.jit, static_argnums=(0,))
    def _compute_energy_masked(self, distance, mask):
        """
        Compute the LJ energy based on an array representing the distances between a given particle and its neighbors.
        Since the distance array is padded to a fixed length, we need to mask out the padded values before summing the energy.

        Parameters
        ----------
        distance : jnp.array
            The distances between a given particle and its neighbors
        mask : jnp.array
            An array indicating which values in the distance array are valid and which are padded [1.0 or 0.0]
        """

        # we can just multiply by the mask rather than using jnp.where to mask.
        energy = mask * (
            4
            * self.epsilon
            * ((self.sigma / distance) ** 12 - (self.sigma / distance) ** 6)
        )
        return energy.sum()

    def compute_energy(self, positions: jnp.array, nbr_list=None, debug_mode=False):
        """
        Compute the LJ energy.

        Parameters
        ----------
        positions : jnp.array
            The positions of the particles in the system
        nbr_list : NeighborList, default=None
            Instance of a neighbor list or pair list class to use.
            If None, an unoptimized N^2 pairlist will be used without PBC conditions.
        Returns
        -------
        potential_energy : float
            The total potential energy of the system.

        """
        # Compute the pair distances and displacement vectors

        if nbr_list is None:
            log.debug(
                "nbr_list is None, computing  using inefficient N^2 pairlist without PBC."
            )
            # Compute the pairlist for a given set of positions and a cutoff distance
            # Note in this case, we do not need the pairs or displacement vectors
            # Since we already calculate the distance in the pairlist computation
            # Pairs and displacement vectors are needed for an analytical evaluation of the force
            # which we will do as part of testing
            distances, displacement_vectors, pairs = self.compute_pairlist(
                positions, self.cutoff
            )
            # if our pairlist is empty, the particles are non-interacting and
            # the energy will be 0
            if distances.shape[0] == 0:
                return 0.0

            potential_energy = (
                4
                * self.epsilon
                * ((self.sigma / distances) ** 12 - (self.sigma / distances) ** 6)
            )
            # sum over all pairs to get the total potential energy
            return potential_energy.sum()

        else:
            # ensure the neighborlist has been constructed before trying to use it

            if not nbr_list.is_built:
                raise ValueError("Neighborlist must be built before use")

            # ensure that the cutoff in the neighbor list is the same as the cutoff in the potential
            if nbr_list.cutoff != self.cutoff:
                raise ValueError(
                    f"Neighborlist cutoff ({nbr_list.cutoff}) must be the same as the potential cutoff ({self.cutoff})"
                )

            n_neighbors, pairs, mask, dist, displacement_vectors = nbr_list.calculate(
                positions
            )

            potential_energy = jax.vmap(self._compute_energy_masked, in_axes=(0))(
                dist, mask.astype(jnp.float32)
            )
            return potential_energy.sum()

    def compute_force(self, positions: jnp.array, nbr_list=None) -> jnp.array:
        """
        Compute the LJ force using the negative of jax.grad.

        Parameters
        ----------
        positions : jnp.array
            The positions of the particles in the system
        nbr_list : NeighborList, optional
            Instance of the neighborlist class to use. By default, set to None, which will use an N^2 pairlist

        Returns
        -------
        force : jnp.array
            The forces on the particles in the system

        """
        # force = -jax.grad(self.compute_energy)(positions, nbr_list)
        # return force
        return super().compute_force(positions, nbr_list=nbr_list)

    def compute_force_analytical(
        self,
        positions: jnp.array,
    ) -> jnp.array:
        """
        Compute the LJ force using the analytical expression for testing purposes.

        Parameters
        ----------
        positions : jnp.array
            The positions of the particles in the system

        Returns
        -------
        force : jnp.array
            The forces on the particles in the system

        """
        dist, displacement_vector, pairs = self.compute_pairlist(positions, self.cutoff)

        forces = (
            24
            * (self.epsilon / (dist * dist))
            * (2 * (self.sigma / dist) ** 12 - (self.sigma / dist) ** 6)
        ).reshape(-1, 1) * displacement_vector

        force_array = jnp.zeros((positions.shape[0], 3))
        for force, p1, p2 in zip(forces, pairs[0], pairs[1]):
            force_array = force_array.at[p1].add(force)
            force_array = force_array.at[p2].add(-force)
        return force_array


class HarmonicOscillatorPotential(NeuralNetworkPotential):
    def __init__(
        self,
        topology: Topology,
        k: unit.Quantity = 1.0 * unit.kilocalories_per_mole / unit.angstrom**2,
        x0: unit.Quantity = 0.0 * unit.angstrom,
        U0: unit.Quantity = 0.0 * unit.kilocalories_per_mole,
    ):
        if not isinstance(topology, Topology):
            if not isinstance(
                topology, property
            ):  # importing from the topology from the model results in it being a property object
                if topology is not None:
                    raise TypeError(
                        f"Topology must be a Topology object or None, type(topology) = {type(topology)}"
                    )
        if not isinstance(k, unit.Quantity):
            raise TypeError(f"k must be a unit.Quantity, type(k) = {type(k)}")
        if not isinstance(x0, unit.Quantity):
            raise TypeError(f"x0 must be a unit.Quantity, type(x0) = {type(x0)}")
        if not isinstance(U0, unit.Quantity):
            raise TypeError(f"U0 must be a unit.Quantity, type(U0) = {type(U0)}")

        if not k.unit.is_compatible(unit.kilocalories_per_mole / unit.angstrom**2):
            raise ValueError(
                f"k must be a unit.Quantity with units of energy per distance squared, k.unit = {k.unit}"
            )
        if not x0.unit.is_compatible(unit.angstrom):
            raise ValueError(
                f"x0 must be a unit.Quantity with units of distance, x0.unit = {x0.unit}"
            )
        if not U0.unit.is_compatible(unit.kilocalories_per_mole):
            raise ValueError(
                f"U0 must be a unit.Quantity with units of energy, U0.unit = {U0.unit}"
            )

        log.info("Initializing HarmonicOscillatorPotential")
        log.info(f"k = {k}")
        log.info(f"x0 = {x0}")
        log.info(f"U0 = {U0}")
        log.info("Energy is calculate: U(x) = (K/2) * ( (x-x0)^2 + y^2 + z^2 ) + U0")
        self.k = jnp.array(
            k.value_in_unit_system(unit.md_unit_system)
        )  # spring constant
        self.x0 = jnp.array(
            x0.value_in_unit_system(unit.md_unit_system)
        )  # equilibrium position
        self.U0 = jnp.array(
            U0.value_in_unit_system(unit.md_unit_system)
        )  # offset potential energy
        self.topology = topology

    def compute_energy(self, positions: jnp.array, nbr_list=None):
        # the functional form is given by U(x) = (K/2) * ( (x-x0)^2 + y^2 + z^2 ) + U0
        # https://github.com/choderalab/openmmtools/blob/main/openmmtools/testsystems.py#L695

        # compute the displacement vectors
        displacement_vectors = positions - self.x0
        # Uue the 3D harmonic oscillator potential to compute the potential energy
        potential_energy = 0.5 * self.k * jnp.sum(displacement_vectors**2) + self.U0
        return potential_energy
