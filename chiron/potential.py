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

    def compute_energy(self, positions):
        # Compute the pair distances and displacement vectors
        pair_distances = pdist(positions)
        displacement_vectors = squareform(pair_distances)
        # Use the model to compute the potential energy
        potential_energy = self.model(displacement_vectors)
        return potential_energy

    def compute_force(self, positions) -> jnp.ndarray:
        # Compute the force as the negative gradient of the potential energy
        force = -jax.grad(self.compute_energy)(positions)
        return force

    def compute_pairlist(self, positions, cutoff) -> jnp.array:
        # Compute the pairlist for a given set of positions and a cutoff distance
        from scipy.spatial.distance import cdist

        pair_distances = cdist(positions, positions)
        pairs = jnp.where(pair_distances < cutoff)

        # exclude case where i ==j and duplicate pairs
        mask = jnp.where(pairs[0] < pairs[1])

        return pairs[0][mask], pairs[1][mask]


class LJPotential(NeuralNetworkPotential):
    def __init__(self, model, **kwargs):
        self.topology = model.potential.topology  # The topology of the system

        sigma: unit.Quantity = (1.0 * unit.kilocalories_per_mole,)
        epsilon: unit.Quantity = 3.350 * unit.angstroms
        assert isinstance(sigma, unit.Quantity)
        assert isinstance(epsilon, unit.Quantity)

        self.sigma = sigma.value_in_unit(
            unit.kilocalories_per_mole
        )  # The distance at which the potential is zero
        self.epsilon = epsilon.value_in_unit(
            unit.angstrom
        )  # The depth of the potential well

    def compute_energy(self, positions: unit.Quantity):
        # Compute the pair distances and displacement vectors
        positions = jnp.array(positions.value_in_unit(unit.angstrom))
        pair_distances = pdist(positions)
        displacement_vectors = squareform(pair_distances)
        # Use the Lennard-Jones potential to compute the potential energy
        potential_energy = (
            4
            * self.epsilon
            * (
                (self.sigma / displacement_vectors) ** 12
                - (self.sigma / displacement_vectors) ** 6
            )
        )
        return potential_energy


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

    def compute_energy(self, positions: jnp.array):
        # the functional form is given by U(x) = (K/2) * ( (x-x0)^2 + y^2 + z^2 ) + U0
        # https://github.com/choderalab/openmmtools/blob/main/openmmtools/testsystems.py#L695

        # compute the displacement vectors
        displacement_vectors = positions[0] - self.x0

        # Uue the 3D harmonic oscillator potential to compute the potential energy
        potential_energy = 0.5 * self.k * jnp.sum(displacement_vectors**2) + self.U0
        return potential_energy
