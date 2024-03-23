# Tests convergence of protocols. This is not intended to be part of the CI GH action tests.
import pytest
import uuid

# check if the test is run on a local machine
import os


@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_convergence")
    return tmpdir


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skip(reason="Tests takes too long")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test takes too long.")
def test_convergence_of_MC_estimator(prep_temp_dir):
    from openmm import unit

    # Initalize the testsystem
    from loguru import logger
    import sys

    logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])

    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology,
        unit.Quantity(1.0, unit.kilocalories_per_mole / unit.angstroms**2),
        U0=ho.U0,
    )

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential,
        temperature=300 * unit.kelvin,
        volume=30 * (unit.angstrom**3),
    )
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = SamplerState(
        positions=ho.positions, current_PRNG_key=PRNG.get_random_key()
    )

    from chiron.reporters import _SimulationReporter

    id = uuid.uuid4()

    simulation_reporter = _SimulationReporter(f"{prep_temp_dir}/test_{id}.h5")

    # Initalize the move set (here only LangevinDynamicsMove)
    from chiron.mcmc import MonteCarloDisplacementMove, MoveSchedule, MCMCSampler

    mc_displacement_move = MonteCarloDisplacementMove(
        number_of_moves=1_000,
        displacement_sigma=0.5 * unit.angstrom,
        atom_subset=[0],
        reporter=simulation_reporter,
    )

    move_set = MoveSchedule([("MonteCarloDisplacementMove", mc_displacement_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=5)  # how many times to repeat

    # Check if estimates are close to the expected value
    import matplotlib.pyplot as plt
    from openmm import unit

    chiron_energy = (
        simulation_reporter.get_property("energy") * unit.kilojoule_per_mole
    ).value_in_unit_system(unit.md_unit_system)
    plt.plot(chiron_energy)

    print("Expectation values generated with chiron")
    import jax.numpy as jnp

    es = jnp.array(chiron_energy)
    print(es.mean(), es.std())

    print("Expectation values from openmmtools")

    class State:
        def __init__(self, temperature):
            self.temperature = temperature

    print(
        ho.get_potential_expectation(State(300 * unit.kelvin)),
        ho.get_potential_standard_deviation(State(300 * unit.kelvin)),
    )
    import jax.numpy as jnp

    jnp.allclose(
        es.mean(),
        ho.get_potential_expectation(State(300 * unit.kelvin)).value_in_unit_system(
            unit.md_unit_system
        ),
        atol=0.1,
    )
    jnp.allclose(
        es.std(),
        ho.get_potential_standard_deviation(
            State(300 * unit.kelvin)
        ).value_in_unit_system(unit.md_unit_system),
        atol=0.1,
    )


@pytest.mark.skip(reason="Tests takes too long")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test takes too long.")
def test_langevin_dynamics_with_LJ_fluid(prep_temp_dir):
    from chiron.integrators import LangevinIntegrator
    from chiron.states import SamplerState, ThermodynamicState
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace
    from openmm import unit
    from openmmtools.testsystems import LennardJonesFluid

    lj_fluid = LennardJonesFluid(reduced_density=0.1, nparticles=5000)
    # initialize potential
    from chiron.potential import LJPotential

    cutoff = 3 * 0.34 * unit.nanometer
    skin = 0.5 * unit.nanometer
    lj_potential = LJPotential(
        lj_fluid.topology, sigma=0.34 * unit.nanometer, cutoff=cutoff
    )

    print(lj_fluid.system.getDefaultPeriodicBoxVectors())
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = SamplerState(
        positions=lj_fluid.positions,
        box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors(),
        current_PRNG_key=PRNG.get_random_key(),
    )
    print(sampler_state.positions.shape)
    print(sampler_state.box_vectors)

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
    )
    nbr_list.build_from_state(sampler_state)

    # initialize states and integrator
    from chiron.integrators import LangevinIntegrator

    thermodynamic_state = ThermodynamicState(
        potential=lj_potential, temperature=300 * unit.kelvin
    )

    from chiron.reporters import LangevinDynamicsReporter

    id = uuid.uuid4()
    reporter = LangevinDynamicsReporter(f"{prep_temp_dir}/test_{id}.h5")

    integrator = LangevinIntegrator(reporter=reporter, report_interval=100)
    integrator.run(
        sampler_state,
        thermodynamic_state,
        number_of_steps=1000,
        nbr_list=nbr_list,
        progress_bar=True,
    )


@pytest.mark.skip(reason="Tests takes too long")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test takes too long.")
def test_ideal_gas(prep_temp_dir):
    from openmmtools.testsystems import IdealGas
    from openmm import unit

    n_particles = 216
    temperature = 298 * unit.kelvin
    pressure = 1 * unit.atmosphere
    mass = unit.Quantity(39.9, unit.gram / unit.mole)

    ideal_gas = IdealGas(
        nparticles=n_particles, temperature=temperature, pressure=pressure
    )

    from chiron.potential import IdealGasPotential
    from chiron.utils import PRNG
    import jax.numpy as jnp

    #
    cutoff = 0.0 * unit.nanometer
    ideal_gas_potential = IdealGasPotential(ideal_gas.topology)

    from chiron.states import SamplerState, ThermodynamicState

    # define the thermodynamic state
    thermodynamic_state = ThermodynamicState(
        potential=ideal_gas_potential,
        temperature=temperature,
        pressure=pressure,
    )

    PRNG.set_seed(1234)

    # define the sampler state
    sampler_state = SamplerState(
        positions=ideal_gas.positions,
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=ideal_gas.system.getDefaultPeriodicBoxVectors(),
    )

    from chiron.neighbors import PairListNsqrd, OrthogonalPeriodicSpace

    # define the pair list for an orthogonal periodic space
    # since particles are non-interacting, this will not really do much
    # but will appropriately wrap particles in space
    nbr_list = PairListNsqrd(OrthogonalPeriodicSpace(), cutoff=cutoff)
    nbr_list.build_from_state(sampler_state)

    from chiron.reporters import MCReporter

    # initialize a reporter to save the simulation data
    filename = "test_mc_ideal_gas.h5"
    import os

    if os.path.isfile(filename):
        os.remove(filename)
    reporter = MCReporter(filename, 1)

    from chiron.mcmc import (
        MonteCarloDisplacementMove,
        MonteCarloBarostatMove,
        MoveSchedule,
        MCMCSampler,
    )

    mc_displacement_move = MonteCarloDisplacementMove(
        displacement_sigma=0.1 * unit.nanometer,
        number_of_moves=10,
        reporter=reporter,
        autotune=True,
        autotune_interval=100,
    )

    mc_barostat_move = MonteCarloBarostatMove(
        volume_max_scale=0.2,
        number_of_moves=100,
        reporter=reporter,
        autotune=True,
        autotune_interval=100,
    )
    move_set = MoveSchedule(
        [
            ("MonteCarloDisplacementMove", mc_displacement_move),
            ("MonteCarloBarostatMove", mc_barostat_move),
        ]
    )

    sampler = MCMCSampler(move_set)
    sampler.run(
        sampler_state, thermodynamic_state, n_iterations=10, nbr_list=nbr_list
    )  # how many times to repeat

    volume = reporter.get_property("volume")

    # get expectations
    ideal_volume = ideal_gas.get_volume_expectation(thermodynamic_state)
    ideal_volume_std = ideal_gas.get_volume_standard_deviation(thermodynamic_state)

    print(ideal_volume, ideal_volume_std)

    volume_mean = jnp.mean(jnp.array(volume)) * unit.nanometer**3
    volume_std = jnp.std(jnp.array(volume)) * unit.nanometer**3

    print(volume_mean, volume_std)

    ideal_density = mass * n_particles / unit.AVOGADRO_CONSTANT_NA / ideal_volume
    measured_density = mass * n_particles / unit.AVOGADRO_CONSTANT_NA / volume_mean

    assert jnp.isclose(
        ideal_density.value_in_unit(unit.kilogram / unit.meter**3),
        measured_density.value_in_unit(unit.kilogram / unit.meter**3),
        atol=1e-1,
    )
    # see if within 5% of ideal volume
    assert abs(ideal_volume - volume_mean) / ideal_volume < 0.05

    # see if within 10% of the ideal standard deviation of the volume
    assert abs(ideal_volume_std - volume_std) / ideal_volume_std < 0.1
