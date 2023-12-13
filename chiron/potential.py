import jax
import jax.numpy as jnp
from scipy.spatial.distance import pdist, squareform
from loguru import logger as log
from openmm import unit
from openmm.app import Topology
import jax.numpy as jnp


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
        # Compute the pairlist for a given set of positions and a cutoff distance
        # Need to replace this with jax compatible code
        # from scipy.spatial.distance import cdist
        #
        # pair_distances = cdist(positions, positions)
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
        sigma: unit.Quantity = 3.350 * unit.angstroms,
        epsilon: unit.Quantity = 1.0 * unit.kilocalories_per_mole,
        cutoff: unit.Quantity = unit.Quantity(1.0, unit.nanometer),

    ):

        assert isinstance(sigma, unit.Quantity)
        assert isinstance(epsilon, unit.Quantity)

        self.sigma = sigma.value_in_unit_system(
            unit.md_unit_system
        )  # The distance at which the potential is zero
        self.epsilon = epsilon.value_in_unit_system(
            unit.md_unit_system
        )  # The depth of the potential well
        # The cutoff for a potential is often linked with the parameters and isn't really
        # something I think we should be changing dynamically.
        self.cutoff = cutoff.value_in_unit_system(unit.md_unit_system)

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
    def compute_energy(
        self,
        positions: jnp.array,
        nbr_list=None,
    ):
        """
        Compute the LJ energy.

        Parameters
        ----------
        positions : jnp.array
            The positions of the particles in the system
        nbr_list : NeighborList, optional
            Instance of the neighborlist class to use. By default, set to None, which will use an N^2 pairlist

        Returns
        -------
        potential_energy : float
            The total potential energy of the system.

        """
        # Compute the pair distances and displacement vectors

        if nbr_list is None:
            # Compute the pairlist for a given set of positions and a cutoff distance
            # Note in this case, we do not need the pairs or displacement vectors
            # Since we already calculate the distance in the pairlist computation
            # Pairs and displacement vectors are needed for an analytical evaluation of the force
            # which we will do as part of testing
            distances, displacement_vectors, pairs  = self.compute_pairlist(positions, self.cutoff)
            # if our pairlist is empty, the particles are non-interacting and
            # the energy will be 0
            if distances.shape[0] == 0:
                return 0.0

            potential_energy = (
                    4 * self.epsilon * ((self.sigma / distances) ** 12 - (self.sigma / distances) ** 6)
            )
            # sum over all pairs to get the total potential energy
            return potential_energy.sum()

        else:
            # ensure the neighborlist has been constructed before trying to use it
            assert(nbr_list.is_built)
            # ensure that the cutoff in the neighbor list is the same as the cutoff in the potential
            assert(nbr_list.cutoff == self.cutoff)

            n_neighbors, mask, dist, displacement_vectors = nbr_list.calculate(positions)
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
        #force = -jax.grad(self.compute_energy)(positions, nbr_list)
        #return force
        return super().compute_force(positions, nbr_list=nbr_list)

    def compute_force_analytical(self, positions: jnp.array, nbr_list=None) -> jnp.array:
        """
        Compute the LJ force using the analytical expression.

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
        if nbr_list is None:
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
        assert isinstance(k, unit.Quantity)
        assert isinstance(x0, unit.Quantity)
        assert isinstance(U0, unit.Quantity)
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

