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
        distances, displacement_vectors, pairlist = nn_potential.compute_pairlist(
            positions, cutoff
        )
        assert pairlist[0].size == expected[0] and pairlist[1].size == expected[1]

    # Test with ethanol molecule
    pdb_file = get_data_file_path("ethanol.pdb")
    pdb = app.PDBFile(pdb_file)
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True).value_in_unit_system(unit.md_unit_system)

    # Test compute_pairlist method
    cutoff = 0.2
    distances, displacement_vectors, pairlist = nn_potential.compute_pairlist(
        positions, cutoff
    )
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


def test_harmonic_oscillator_input_checking():
    # topology check
    with pytest.raises(TypeError):
        HarmonicOscillatorPotential(1)
    with pytest.raises(TypeError):
        HarmonicOscillatorPotential(None, k=1.0)
    with pytest.raises(TypeError):
        HarmonicOscillatorPotential(None, x0=1.0)
    with pytest.raises(TypeError):
        HarmonicOscillatorPotential(None, U0=1.0)

    with pytest.raises(ValueError):
        HarmonicOscillatorPotential(None, k=1.0 * unit.nanometer)
    with pytest.raises(ValueError):
        HarmonicOscillatorPotential(None, x0=1.0 * unit.kilocalories_per_mole)
    with pytest.raises(ValueError):
        HarmonicOscillatorPotential(None, U0=1.0 * unit.nanometer)


def test_lj_input_checking():
    # topology check
    with pytest.raises(TypeError):
        LJPotential(1)
    with pytest.raises(TypeError):
        LJPotential(None, sigma=1.0)
    with pytest.raises(TypeError):
        LJPotential(None, epsilon=1.0)
    with pytest.raises(TypeError):
        LJPotential(None, cutoff=1.0)

    with pytest.raises(ValueError):
        LJPotential(None, sigma=1.0 * unit.kilocalories_per_mole)
    with pytest.raises(ValueError):
        LJPotential(None, epsilon=1.0 * unit.nanometer)
    with pytest.raises(ValueError):
        LJPotential(None, cutoff=1.0 * unit.kilocalories_per_mole)

    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

    positions = jnp.array([[0, 0, 0], [1, 0, 0]])
    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    lj = LJPotential(None, sigma=1.0 * unit.nanometer)
    nbr_list = NeighborListNsqrd(OrthogonalPeriodicSpace(), cutoff=2.0 * unit.nanometer)

    # capture the error associated with not building the neighborlist
    with pytest.raises(ValueError):
        lj.compute_energy(positions, nbr_list)
    nbr_list.build(positions, box_vectors)

    # capture the error associated cutoffs not matching
    with pytest.raises(ValueError):
        lj.compute_energy(positions, nbr_list)


def test_lennard_jones():
    # This will evaluate two LJ particles to ensure the energy and force are correct
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace
    from chiron.states import SamplerState

    sigma = 1.0
    epsilon = 1.0
    cutoff = 3.0
    skin = 0.5
    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    space = OrthogonalPeriodicSpace()

    lj_pot = LJPotential(
        None,
        unit.Quantity(sigma, unit.nanometer),
        unit.Quantity(epsilon, unit.kilojoules_per_mole),
        unit.Quantity(cutoff, unit.nanometer),
    )
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    for i in range(1, 11):
        positions = jnp.array([[0, 0, 0], [i * 0.25 * 2 ** (1 / 6), 0, 0]])

        state = SamplerState(
            positions=unit.Quantity(positions, unit.nanometer),
            current_PRNG_key=PRNG.get_random_key(),
            box_vectors=unit.Quantity(box_vectors, unit.nanometer),
        )
        nbr_list = NeighborListNsqrd(
            space,
            cutoff=unit.Quantity(cutoff, unit.nanometer),
            skin=unit.Quantity(skin, unit.nanometer),
            n_max_neighbors=5,
        )
        nbr_list.build_from_state(state)
        # first use the pairlist
        energy_chiron = lj_pot.compute_energy(positions)
        energy_chiron_nbr = lj_pot.compute_energy(positions, nbr_list)

        displacement_vector = positions[0] - positions[1]
        dist = jnp.linalg.norm(displacement_vector)

        energy_analytical = 4.0 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)

        assert jnp.isclose(
            energy_chiron, energy_analytical
        ), "Energy from chiron using a pair list does not match the analytical energy calculation"
        assert jnp.isclose(
            energy_chiron_nbr, energy_analytical
        ), "Energy from chiron using a neighbor list does not match the analytical energy calculation"

        force_chiron = lj_pot.compute_force(positions)
        force_chiron_nbr = lj_pot.compute_force(positions, nbr_list)

        # this uses the pairlist to calculate the analytical force
        force_chiron_analytical = lj_pot.compute_force_analytical(positions)

        force = (
            24
            * (epsilon / (dist * dist))
            * (2 * (sigma / dist) ** 12 - (sigma / dist) ** 6)
        ) * displacement_vector

        forces_analytical = jnp.array([force, -force])

        assert jnp.allclose(
            force_chiron, forces_analytical, atol=1e-5
        ), "Force from chiron using pair list does not match analytical force"
        assert jnp.allclose(
            force_chiron_nbr, forces_analytical, atol=1e-5
        ), "Force from chiron using neighbor list does not match analytical force"
        assert jnp.allclose(
            force_chiron_analytical, forces_analytical, atol=1e-5
        ), "Force from chiron analytical using pair list does not match analytical force"
