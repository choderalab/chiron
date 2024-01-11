import pytest
import uuid


@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_mcmc")
    return tmpdir


def test_sample_from_harmonic_osciallator(prep_temp_dir):
    """
    Test sampling from a harmonic oscillator using local moves.

    This test initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses a Langevin integrator to sample
    from the oscillator's state space.
    """
    from openmm import unit

    # initialize openmmtestsystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # initialze HO potential
    from chiron.potential import HarmonicOscillatorPotential

    # NOTE: let's construct this potential from the openmmtools test system in the future
    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # intialize the states
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        potential=harmonic_potential, temperature=300 * unit.kelvin
    )
    sampler_state = SamplerState(x0=ho.positions)
    from chiron.integrators import LangevinIntegrator

    from chiron.reporters import LangevinDynamicsReporter, BaseReporter

    id = uuid.uuid4()
    wd = prep_temp_dir.join(f"_test_{id}")
    BaseReporter.set_directory(wd)
    reporter = LangevinDynamicsReporter()

    integrator = LangevinIntegrator(
        stepsize=2 * unit.femtosecond, reporter=reporter, save_frequency=1
    )

    integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=5,
    )
    integrator.reporter.close()
    import jax.numpy as jnp
    import h5py

    h5 = h5py.File(f"{wd}/{LangevinDynamicsReporter.get_name()}.h5", "r")
    keys = h5.keys()

    assert "energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" in keys, "Traj not in keys"

    energy = h5["energy"][:]
    print(energy)

    reference_energy = jnp.array(
        [0.01984119, 0.08067884, 0.17772843, 0.30644223, 0.4665345]
    )
    assert jnp.allclose(energy, reference_energy)


def test_sample_from_harmonic_osciallator_with_MCMC_classes_and_LangevinDynamics(
    prep_temp_dir,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Langevin dynamics.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Langevin dynamics move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.mcmc import LangevinDynamicsMove, MoveSchedule, MCMCSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillatorArray

    ho = HarmonicOscillatorArray()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential,
        temperature=300 * unit.kelvin,
        volume=30 * (unit.angstrom**3),
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set (here only LangevinDynamicsMove) and reporter
    from chiron.reporters import LangevinDynamicsReporter, BaseReporter

    BaseReporter.set_directory(prep_temp_dir)

    simulation_reporter = LangevinDynamicsReporter(1)
    langevin_move = LangevinDynamicsMove(
        nr_of_steps=10, seed=1234, reporter=simulation_reporter
    )

    move_set = MoveSchedule([("LangevinMove", langevin_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_sample_from_harmonic_osciallator_with_MCMC_classes_and_MetropolisDisplacementMove(
    prep_temp_dir,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Metropolis displacement move.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Metropolis displacement move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.mcmc import MetropolisDisplacementMove, MoveSchedule, MCMCSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential,
        temperature=300 * unit.kelvin,
        volume=30 * (unit.angstrom**3),
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set and reporter
    from chiron.reporters import MCReporter, BaseReporter

    wd = prep_temp_dir.join(f"_test_{uuid.uuid4()}")
    BaseReporter.set_directory(wd)
    simulation_reporter = MCReporter(1)

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=[0],
        reporter=simulation_reporter,
    )

    move_set = MoveSchedule([("MetropolisDisplacementMove", mc_displacement_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_sample_from_harmonic_osciallator_array_with_MCMC_classes_and_MetropolisDisplacementMove(
    prep_temp_dir,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Metropolis displacement move.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Metropolis displacement move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.mcmc import MetropolisDisplacementMove, MoveSchedule, MCMCSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillatorArray

    ho = HarmonicOscillatorArray()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential,
        temperature=300 * unit.kelvin,
        volume=30 * (unit.angstrom**3),
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set and reporter
    from chiron.reporters import MCReporter, BaseReporter

    wd = prep_temp_dir.join(f"_test_{uuid.uuid4()}")
    BaseReporter.set_directory(wd)

    simulation_reporter = MCReporter(1)

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=None,
        reporter=simulation_reporter,
    )

    move_set = MoveSchedule([("MetropolisDisplacementMove", mc_displacement_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_thermodynamic_state_inputs():
    from chiron.states import ThermodynamicState
    from openmm import unit
    from openmmtools.testsystems import HarmonicOscillatorArray

    ho = HarmonicOscillatorArray()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K)

    with pytest.raises(TypeError):
        ThermodynamicState(potential=harmonic_potential, temperature=300)

    with pytest.raises(ValueError):
        ThermodynamicState(
            potential=harmonic_potential, temperature=300 * unit.angstrom
        )

    ThermodynamicState(potential=harmonic_potential, temperature=300 * unit.kelvin)

    with pytest.raises(TypeError):
        ThermodynamicState(potential=harmonic_potential, volume=1000)
    with pytest.raises(ValueError):
        ThermodynamicState(potential=harmonic_potential, volume=1000 * unit.kelvin)

    ThermodynamicState(potential=harmonic_potential, volume=1000 * (unit.angstrom**3))

    with pytest.raises(TypeError):
        ThermodynamicState(potential=harmonic_potential, pressure=100)

    with pytest.raises(ValueError):
        ThermodynamicState(potential=harmonic_potential, pressure=100 * unit.kelvin)

    ThermodynamicState(potential=harmonic_potential, pressure=100 * unit.atmosphere)


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
