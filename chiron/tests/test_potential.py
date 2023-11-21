import jax.numpy as np
import pytest
from chiron.potential import (
    NeuralNetworkPotential,
    HarmonicOscillatorPotential,
    LJPotential,
)
from chiron.topology import Topology


# Test NeuralNetworkPotential
def test_neural_network_pairlist():
    from chiron.topology import Atom

    # Create a topology object
    topology = Topology([Atom(atomic_number=1), Atom(atomic_number=1)])

    # Create a neural network potential object
    nn_potential = NeuralNetworkPotential(model=None, topology=topology)

    # Test compute_energy method
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # Test compute_pairlist method
    cutoff = 2.0
    pairlist = nn_potential.compute_pairlist(positions, cutoff)
    print(pairlist)
    assert pairlist.size == 1  # there is one pair that is within the cutoff distance

    # Test compute_pairlist method
    cutoff = 1.5
    pairlist = nn_potential.compute_pairlist(positions, cutoff)
    print(pairlist)
    assert pairlist.size == 0  # there is no pair that is within the cutoff distance


# Test LJPotential
def test_lj_potential():
    from chiron.topology import Atom
    from openmm import unit

    # Create a topology object
    topology = Topology([Atom(atomic_number=1), Atom(atomic_number=1)])

    # Create an LJ potential object
    sigma = 1.0 * unit.kilocalories_per_mole
    epsilon = 3.0 * unit.angstroms
    lj_potential = LJPotential(topology, sigma, epsilon)

    # Test compute_energy method
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]) * unit.angstrom
    energy = lj_potential.compute_energy(positions)
    assert isinstance(energy, float)

    # Test compute_force method
    forces = lj_potential.compute_force(positions)
    assert forces.shape == positions.shape


# Test HarmonicOscillatorPotential
def test_harmonic_oscillator_potential():
    # Create a topology object
    topology = Topology()

    # Create a harmonic oscillator potential object
    k = 1.0
    x0 = np.array([0.0, 0.0, 0.0])
    U0 = 0.0
    harmonic_potential = HarmonicOscillatorPotential(topology, k, x0, U0)

    # Test compute_energy method
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    energy = harmonic_potential.compute_energy(positions)
    assert isinstance(energy, float)

    # Test compute_force method
    forces = harmonic_potential.compute_force(positions)
    assert forces.shape == positions.shape


# Run the tests
if __name__ == "__main__":
    pytest.main()
