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

    from chiron.reporters import SimulationReporter

    id = uuid.uuid4()
    h5_file = f"test_{id}.h5"
    reporter = SimulationReporter(f"{prep_temp_dir}/{h5_file}", 1)

    integrator = LangevinIntegrator(
        stepsize=0.2 * unit.femtosecond, reporter=reporter, save_frequency=1
    )

    r = integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=5,
    )

    import jax.numpy as jnp
    import h5py

    h5 = h5py.File(f"{prep_temp_dir}/{h5_file}", "r")
    keys = h5.keys()

    assert "energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" in keys, "Traj not in keys"

    energy = h5["energy"][:]
    print(energy)

    reference_energy = jnp.array(
        [0.00019308, 0.00077772, 0.00174247, 0.00307798, 0.00479007]
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
    from chiron.mcmc import LangevinDynamicsMove, MoveSet, MCMCSampler

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
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter(
        f"{prep_temp_dir}/test_{uuid.uuid4()}.h5", None, 1
    )
    langevin_move = LangevinDynamicsMove(
        nr_of_steps=10, seed=1234, simulation_reporter=simulation_reporter
    )

    move_set = MoveSet([("LangevinMove", langevin_move)])

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
    from chiron.mcmc import MetropolisDisplacementMove, MoveSet, MCMCSampler

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
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter(
        f"{prep_temp_dir}/test_{uuid.uuid4()}.h5", 1
    )

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=[0],
        simulation_reporter=simulation_reporter,
    )

    move_set = MoveSet([("MetropolisDisplacementMove", mc_displacement_move)])

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
    from chiron.mcmc import MetropolisDisplacementMove, MoveSet, MCMCSampler

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
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter(
        f"{prep_temp_dir}/test_{uuid.uuid4()}.h5", 1
    )

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=None,
        simulation_reporter=simulation_reporter,
    )

    move_set = MoveSet([("MetropolisDisplacementMove", mc_displacement_move)])

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


def test_mc_barostat_setting():
    import jax.numpy as jnp
    from chiron.mcmc import MCBarostatMove

    barostat_move = MCBarostatMove(
        seed=1234,
        volume_max_scale=0.01,
        nr_of_moves=2,
    )

    assert barostat_move.volume_max_scale == 0.01

    from chiron.potential import LJPotential
    from openmm import unit

    sigma = 0.34 * unit.nanometer
    epsilon = 0.238 * unit.kilocalories_per_mole
    cutoff = 3.0 * sigma

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

    from openmm.app import Topology, Element

    topology = Topology()
    element = Element.getBySymbol("Ar")
    chain = topology.addChain()
    residue = topology.addResidue("system", chain)
    for i in range(positions.shape[0]):
        topology.addAtom("Ar", element, residue)

    lj_potential = LJPotential(topology, sigma=sigma, epsilon=epsilon, cutoff=cutoff)

    from chiron.states import SamplerState, ThermodynamicState

    # define the sampler state
    sampler_state = SamplerState(
        x0=positions,
        box_vectors=box_vectors,
    )

    # define the thermodynamic state
    thermodynamic_state = ThermodynamicState(
        potential=lj_potential,
        temperature=300 * unit.kelvin,
        pressure=1.0 * unit.atmosphere,
    )

    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

    # define the neighbor list for an orthogonal periodic space
    skin = 0.5 * unit.nanometer

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
    )

    barostat_move.run(sampler_state, thermodynamic_state, nbr_list, True)

    assert barostat_move.statistics["n_accepted"] == 1
    assert barostat_move.statistics["n_proposed"] == 2

    assert jnp.all(
        sampler_state.x0
        == jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.99709356, 0.0, 0.0],
                [0.0, 0.99709356, 0.0],
                [0.0, 0.0, 0.99709356],
                [0.99709356, 0.99709356, 0.0],
                [0.99709356, 0.0, 0.99709356],
                [0.0, 0.99709356, 0.99709356],
                [0.99709356, 0.99709356, 0.99709356],
            ]
        )
    )
    assert jnp.all(
        sampler_state.box_vectors
        == jnp.array([[9.987228, 0.0, 0.0], [0.0, 9.987228, 0.0], [0.0, 0.0, 9.987228]])
    )


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
