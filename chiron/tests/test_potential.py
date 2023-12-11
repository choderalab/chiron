import jax.numpy as jnp
import pytest
from chiron.potential import (
    NeuralNetworkPotential,
    HarmonicOscillatorPotential,
    LJPotential,
)

from chiron.utils import get_data_file_path
from openmm import unit

from openmm import app


# Test NeuralNetworkPotential
def test_neural_network_pairlist():
    """
    Test the pairlist computation for a NeuralNetworkPotential.

    This function tests the compute_pairlist method for different cutoff distances
    using a simple two-particle system and ethanol molecule.
    """

    pdb_file = get_data_file_path("two_particles_1.pdb")
    pdb = app.PDBFile(pdb_file)
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system)

    # Create a neural network potential object
    nn_potential = NeuralNetworkPotential(model=None, topology=topology)

    # Test with different cutoffs
    cutoffs = [0.2, 0.1]
    expected_pairs = [(1, 1), (0, 0)]
    for cutoff, expected in zip(cutoffs, expected_pairs):
        pairlist = nn_potential.compute_pairlist(positions, cutoff)
        assert pairlist[0].size == expected[0] and pairlist[1].size == expected[1]

    # Test with ethanol molecule
    pdb_file = get_data_file_path("ethanol.pdb")
    pdb = app.PDBFile(pdb_file)
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system)

    # Test compute_pairlist method
    cutoff = 0.2
    pairlist = nn_potential.compute_pairlist(positions, cutoff)
    print(pairlist)
    assert (
        pairlist[0].size == 12 and pairlist[1].size == 12
    )  # there are 12 pairs within the cutoff distance


# Test HarmonicOscillatorPotential
def test_harmonic_oscillator_potential():
    """
    Test the energy computation of a HarmonicOscillatorPotential.

    This function verifies the energy computed by a HarmonicOscillatorPotential for
    various positions, comparing the computed energies with expected values.
    """
    k = 100.0 * unit.kilocalories_per_mole / unit.angstroms**2
    U0 = 0.0 * unit.kilocalories_per_mole
    x0 = unit.Quantity(jnp.array([[0.0, 0.0, 0.0]]), unit.angstrom)

    from openmmtools.testsystems import HarmonicOscillator as ho

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, k, x0, U0)
    test_positions = [
        jnp.array([[0.0, 0.0, 0.0]]) * unit.angstrom,
        jnp.array([[0.2, 0.2, 0.2]]) * unit.angstrom,
        jnp.array([[0.2, 0.0, 0.0]]) * unit.angstrom,
        jnp.array([[-0.2, 0.0, 0.0]]) * unit.angstrom,
        jnp.array([[-0.0, 0.2, 0.0]]) * unit.angstrom,
    ]
    expected_energies = [
        0.0,
        25.10400390625,
        8.368000984191895,
        8.368000984191895,
        8.368000984191895,
    ]

    for pos, expected_energy in zip(test_positions, expected_energies):
        positions_without_unit = jnp.array(
            pos.value_in_unit_system(unit.md_unit_system)
        )
        energy = float(harmonic_potential.compute_energy(positions_without_unit))
        assert jnp.isclose(energy, expected_energy), f"Energy at {pos} is incorrect."

    # Test compute_force method
    forces = harmonic_potential.compute_force(positions_without_unit)
    assert forces.shape == positions_without_unit.shape, "Forces shape mismatch."
