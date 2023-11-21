import jax
import jax.numpy as np
from scipy.spatial.distance import pdist, squareform
from .topology import Topology
from loguru import logger as log
from openmm import unit


class NeuralNetworkPotential:
    def __init__(self, model, topology: Topology):
        self.topology = topology  # The topology of the system
        if model is None:
            log.warning("No model provided, using default model")
        else:
            self.model = model(self.topology)  # The trained neural network model

    def compute_energy(self, positions):
        # Compute the pair distances and displacement vectors
        pair_distances = pdist(positions)
        displacement_vectors = squareform(pair_distances)
        # Use the model to compute the potential energy
        potential_energy = self.model(displacement_vectors)
        return potential_energy

    def compute_force(self, positions):
        # Compute the force as the negative gradient of the potential energy
        force = -jax.grad(self.compute_energy)(positions)
        return force

    def compute_pairlist(self, positions, cutoff) -> np.array:
        # Compute the pairlist for a given set of positions and a cutoff distance
        pair_distances = pdist(positions)
        pairlist = np.where(pair_distances < cutoff)
        return pairlist[0]


class LJPotential(NeuralNetworkPotential):
    def __init__(
        self,
        topology: Topology,
        sigma: unit.Quantity = 1.0 * unit.kilocalories_per_mole,
        epsilon: unit.Quantity = 3.350 * unit.angstroms,
    ):
        assert isinstance(sigma, unit.Quantity)
        assert isinstance(epsilon, unit.Quantity)

        self.sigma = sigma.value_in_unit(
            unit.kilocalories_per_mole
        )  # The distance at which the potential is zero
        self.epsilon = epsilon.value_in_unit(
            unit.angstrom
        )  # The depth of the potential well
        self.topology = topology  # The topology of the system

    def compute_energy(self, positions: unit.Quantity):
        # Compute the pair distances and displacement vectors
        positions = positions.value_in_unit(unit.angstrom)
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

    def compute_force(self, positions):
        # Compute the force as the negative gradient of the potential energy
        force = -jax.grad(self.compute_energy)(positions)
        return force


class HarmonicOscillatorPotential(NeuralNetworkPotential):
    def __init__(self, topology, k, x0, U0):
        self.topology = topology
        self.k = k  # spring constant
        self.x0 = x0  # equilibrium position
        self.U0 = U0  # offset potential energy

    def compute_energy(self, positions):
        # the functional form is given by U(x) = (K/2) * ( (x-x0)^2 + y^2 + z^2 ) + U0
        # https://github.com/choderalab/openmmtools/blob/main/openmmtools/testsystems.py#L695

        # compute the displacement vectors
        displacement_vectors = positions - self.x0
        # Uue the 3D harmonic oscillator potential to compute the potential energy
        potential_energy = 0.5 * self.k * np.sum(displacement_vectors**2) + self.U0
        return potential_energy
