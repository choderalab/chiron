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
    # Create a topology object
    pdb_file = get_data_file_path("two_particles_1.pdb")
    pdb = app.PDBFile(pdb_file)
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system)
    # Create a neural network potential object
    nn_potential = NeuralNetworkPotential(model=None, topology=topology)

    # Test compute_pairlist method
    cutoff = 0.2
    pairlist = nn_potential.compute_pairlist(positions, cutoff)
    assert (
        pairlist[0].size == 1 and pairlist[1].size == 1
    )  # there is one pair that is within the cutoff distance

    # Test compute_pairlist method
    cutoff = 0.1
    pairlist = nn_potential.compute_pairlist(positions, cutoff)
    assert (
        pairlist[0].size == 0 and pairlist[1].size == 0
    )  # there is no pair that is within the cutoff distance

    # try this with ethanol
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
    # Create a harmonic oscillator potential object
    k = 100.0 * unit.kilocalories_per_mole / unit.angstroms**2
    U0 = 0.0 * unit.kilocalories_per_mole
    x0 = 0.0 * unit.angstrom
    from openmmtools.testsystems import HarmonicOscillator as ho

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, k, x0, U0)
    positions = jnp.array([0.0, 0.0, 0.0]) * unit.angstrom
    # Test compute_energy method
    positions_without_unit = jnp.array(
        positions.value_in_unit_system(unit.md_unit_system)
    )
    energy = float(harmonic_potential.compute_energy(positions_without_unit))
    assert jnp.isclose(energy, 0.0)

    positions = jnp.array([0.2, 0.2, 0.2]) * unit.angstrom
    # Test compute_energy method
    positions_without_unit = jnp.array(
        positions.value_in_unit_system(unit.md_unit_system)
    )
    energy = float(harmonic_potential.compute_energy(positions_without_unit))
    assert jnp.isclose(energy, 25.10400390625)

    positions = jnp.array([0.2, 0.0, 0.0]) * unit.angstrom
    # Test compute_energy method
    positions_without_unit = jnp.array(
        positions.value_in_unit_system(unit.md_unit_system)
    )
    energy = float(harmonic_potential.compute_energy(positions_without_unit))
    assert jnp.isclose(energy, 8.368000984191895)

    positions = jnp.array([-0.2, 0.0, 0.0]) * unit.angstrom
    # Test compute_energy method
    positions_without_unit = jnp.array(
        positions.value_in_unit_system(unit.md_unit_system)
    )
    energy = float(harmonic_potential.compute_energy(positions_without_unit))
    assert jnp.isclose(energy, 8.368000984191895)

    positions = jnp.array([-0.0, 0.2, 0.0]) * unit.angstrom
    # Test compute_energy method
    positions_without_unit = jnp.array(
        positions.value_in_unit_system(unit.md_unit_system)
    )
    energy = float(harmonic_potential.compute_energy(positions_without_unit))
    assert jnp.isclose(energy, 8.368000984191895)

    # Test compute_force method
    forces = harmonic_potential.compute_force(positions_without_unit)
    assert forces.shape == positions_without_unit.shape


# # Test LJPotential
# def test_lj_potential():
#     pdb_file = get_data_file_path("two_particles_1.pdb")
#     pdb = app.PDBFile(pdb_file)
#     topology = pdb.getTopology()
#     positions = pdb.getPositions(asNumpy=True)

#     # Create an LJ potential object
#     sigma = 1.0 * unit.kilocalories_per_mole
#     epsilon = 3.0 * unit.angstroms
#     lj_potential = LJPotential(topology, sigma, epsilon)

#     # Test compute_energy method
#     positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]) * unit.angstrom
#     energy = lj_potential.compute_energy(positions)
#     assert isinstance(energy, float)

#     # Test compute_force method
#     forces = lj_potential.compute_force(positions)
#     assert forces.shape == positions.shape
