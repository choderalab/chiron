import jax
import jax.numpy as np
from scipy.spatial.distance import pdist, squareform
from .topology import Topology


class NeuralNetworkPotential:
    def __init__(self, model, topology: Topology):
        self.topology = topology  # The topology of the system
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

    def compute_pairlist(self, positions, cutoff):
        # Compute the pairlist for a given set of positions and a cutoff distance
        pair_distances = pdist(positions)
        pairlist = np.where(pair_distances < cutoff)
        return pairlist


class DummyNeuralNetworkPotential:
    def __init__(self, sigma, epsilon, topology: Topology):
        self.sigma = sigma  # The distance at which the potential is zero
        self.epsilon = epsilon  # The depth of the potential well
        self.topology = topology  # The topology of the system

    def compute_energy(self, positions):
        # Compute the pair distances and displacement vectors
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
