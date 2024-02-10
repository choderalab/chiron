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
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = SamplerState(
        x0=ho.positions, current_PRNG_key=PRNG.get_random_key()
    )
    from chiron.integrators import LangevinIntegrator

    from chiron.reporters import LangevinDynamicsReporter, BaseReporter

    id = uuid.uuid4()
    wd = prep_temp_dir.join(f"_test_{id}")
    BaseReporter.set_directory(wd)
    reporter = LangevinDynamicsReporter()

    integrator = LangevinIntegrator(
        stepsize=2 * unit.femtosecond,
        reporter=reporter,
        report_frequency=1,
        reinitialize_velocities=True,
    )

    integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=5,
    )
    integrator.reporter.flush_buffer()
    import jax.numpy as jnp
    import h5py

    h5 = h5py.File(f"{wd}/{LangevinDynamicsReporter.get_name()}.h5", "r")
    keys = h5.keys()

    assert "potential_energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" not in keys, "Traj is not in keys"

    energy = h5["potential_energy"][:]
    print(energy)

    reference_energy = jnp.array(
        [0.03551735, 0.1395877, 0.30911613, 0.5495938, 0.85149795]
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
    from chiron.utils import PRNG

    PRNG.set_seed(1234)
    thermodynamic_state = ThermodynamicState(
        harmonic_potential,
        temperature=300 * unit.kelvin,
        volume=30 * (unit.angstrom**3),
    )
    sampler_state = SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())

    # Initalize the move set (here only LangevinDynamicsMove) and reporter
    from chiron.reporters import LangevinDynamicsReporter, BaseReporter

    BaseReporter.set_directory(prep_temp_dir)

    simulation_reporter = LangevinDynamicsReporter(1)

    # the following will reinitialize the velocities for each iteration
    langevin_move = LangevinDynamicsMove(
        nr_of_steps=10, reinitialize_velocities=True, reporter=simulation_reporter
    )

    move_set = MoveSchedule([("LangevinMove", langevin_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(
        sampler_state, thermodynamic_state, n_iterations=2
    )  # how many times to repeat

    # the following will use the initialize velocities function
    from chiron.utils import initialize_velocities

    sampler_state.velocities = initialize_velocities(
        thermodynamic_state.temperature, ho.topology, sampler_state._current_PRNG_key
    )

    langevin_move = LangevinDynamicsMove(
        nr_of_steps=10, reinitialize_velocities=False, reporter=simulation_reporter
    )

    move_set = MoveSchedule([("LangevinMove", langevin_move)])

    # Initalize the sampler
    sampler = MCMCSampler(move_set)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(
        sampler_state, thermodynamic_state, n_iterations=2
    )  # how many times to repeat


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
    from chiron.utils import PRNG

    PRNG.set_seed(1234)
    sampler_state = SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())

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
    sampler = MCMCSampler(move_set)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(
        sampler_state, thermodynamic_state, n_iterations=2
    )  # how many times to repeat


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

    from chiron.utils import PRNG

    PRNG.set_seed(1234)
    sampler_state = SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())

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
    sampler = MCMCSampler(move_set)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(
        sampler_state, thermodynamic_state, n_iterations=2
    )  # how many times to repeat


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


def test_mc_barostat_parameter_setting():
    import jax.numpy as jnp
    from chiron.mcmc import MonteCarloBarostatMove

    barostat_move = MonteCarloBarostatMove(
        volume_max_scale=0.1,
        nr_of_moves=1,
    )

    assert barostat_move.volume_max_scale == 0.1
    assert barostat_move.nr_of_moves == 1


def test_mc_barostat(prep_temp_dir):
    import jax.numpy as jnp

    from chiron.reporters import MCReporter, BaseReporter

    wd = prep_temp_dir.join(f"_test_{uuid.uuid4()}")
    BaseReporter.set_directory(wd)
    simulation_reporter = MCReporter(1)

    from chiron.mcmc import MonteCarloBarostatMove

    barostat_move = MonteCarloBarostatMove(
        volume_max_scale=0.1,
        nr_of_moves=10,
        reporter=simulation_reporter,
        report_frequency=1,
    )

    from chiron.potential import IdealGasPotential
    from openmm import unit

    positions = (
        jnp.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        * unit.nanometer
    )
    box_vectors = (
        jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        * unit.nanometer
    )
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

    from openmm.app import Topology, Element

    topology = Topology()
    element = Element.getBySymbol("Ar")
    chain = topology.addChain()
    residue = topology.addResidue("system", chain)
    for i in range(positions.shape[0]):
        topology.addAtom("Ar", element, residue)

    ideal_gas_potential = IdealGasPotential(topology)

    from chiron.states import SamplerState, ThermodynamicState
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    # define the sampler state
    sampler_state = SamplerState(
        x0=positions, box_vectors=box_vectors, current_PRNG_key=PRNG.get_random_key()
    )

    # define the thermodynamic state
    thermodynamic_state = ThermodynamicState(
        potential=ideal_gas_potential,
        temperature=300 * unit.kelvin,
        pressure=1.0 * unit.atmosphere,
    )

    from chiron.neighbors import PairList, OrthogonalPeriodicSpace

    # since particles are non-interacting and we will not displacece them, the pair list basically
    # does nothing in this case.
    nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=0 * unit.nanometer)

    sampler_state, thermodynamic_state, nbr_list = barostat_move.update(
        sampler_state, thermodynamic_state, nbr_list
    )
    potential_energies = simulation_reporter.get_property("potential_energy")
    volumes = simulation_reporter.get_property("volume")

    # ideal gas treatment, so stored energy will only be a
    # consequence of pressure, volume, and temperature
    from loguru import logger as log

    log.debug(f"PE {potential_energies * unit.kilojoules_per_mole}")
    log.debug(thermodynamic_state.pressure)
    log.debug(thermodynamic_state.beta)
    log.debug(volumes)
    log.debug(volumes * unit.nanometer**3)

    # assert that the PE is always zero
    assert potential_energies[0] == 0
    assert potential_energies[-1] == 0

    # the reduced potential will only be a consequence of the pressure, volume, and temperature

    assert jnp.isclose(
        thermodynamic_state.get_reduced_potential(sampler_state),
        (
            thermodynamic_state.pressure
            * thermodynamic_state.beta
            * (volumes[-1] * unit.nanometer**3)
        ),
        1e-3,
    )

    print(barostat_move.statistics["n_accepted"])
    assert barostat_move.statistics["n_proposed"] == 10
    assert barostat_move.statistics["n_accepted"] == 8


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
