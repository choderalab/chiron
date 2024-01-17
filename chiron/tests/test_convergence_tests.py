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
        harmonic_potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions)

    from chiron.reporters import _SimulationReporter

    id = uuid.uuid4()

    simulation_reporter = _SimulationReporter(f"{prep_temp_dir}/test_{id}.h5")

    # Initalize the move set (here only LangevinDynamicsMove)
    from chiron.mcmc import MetropolisDisplacementMove, MoveSchedule, MCMCSampler

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=100_000,
        displacement_sigma=0.5 * unit.angstrom,
        atom_subset=[0],
        reporter=simulation_reporter,
    )

    move_set = MoveSchedule([("MetropolisDisplacementMove", mc_displacement_move)])

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
    es = chiron_energy
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

    sampler_state = SamplerState(
        x0=lj_fluid.positions,
        box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors(),
    )
    print(sampler_state.x0.shape)
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

    from chiron.reporters import _SimulationReporter

    id = uuid.uuid4()
    reporter = _SimulationReporter(f"{prep_temp_dir}/test_{id}.h5")

    integrator = LangevinIntegrator(reporter=reporter, report_frequency=100)
    integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=2000,
        nbr_list=nbr_list,
        progress_bar=True,
    )
