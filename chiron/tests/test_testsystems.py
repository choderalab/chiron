import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_testsystems")
    return tmpdir


def compute_openmm_reference_energy(testsystem, positions):
    from openmm import unit
    from openmm.app import Simulation
    from openmm import LangevinIntegrator

    system = testsystem.system
    integrator = LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.1 * unit.femtosecond
    )
    sim = Simulation(testsystem.topology, system, integrator)
    sim.context.setPositions(positions)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(e)
    return e


def test_HO():
    """
    Test the harmonic oscillator system using a Langevin integrator.

    This function initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses the Langevin integrator to run the simulation.
    """
    # initialize testystem
    from openmmtools.testsystems import HarmonicOscillator, HarmonicOscillatorArray

    ho = HarmonicOscillator()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    import jax.numpy as jnp
    from openmm import unit

    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology,
        ho.K,
        x0=unit.Quantity(jnp.array([[0.0, 0.0, 0.0]]), unit.angstrom),
        U0=ho.U0,
    )
    ###############################
    # Harmonic Oscillator
    ###############################
    # calculate the energy
    # at the equilibrium position, the energy should be zero
    pos = jnp.array([[0.0, 0.0, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    assert jnp.isclose(e_chiron, 0.0), "Energy at equilibrium position is not zero"

    # calculate the energy
    # off-equilibrium
    pos = jnp.array([[0.1, 0.0, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    # compare with openmm reference for same system
    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"

    pos = jnp.array([[0.0, 0.1, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    # compare with openmm reference for same system
    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"

    ###############################
    # Harmonic OscillatorArray
    ###############################
    ho = HarmonicOscillatorArray()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    import jax.numpy as jnp
    from openmm import unit

    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology,
        ho.K,
        x0=unit.Quantity(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
            unit.nanometer,
        ),
    )

    # calculate the energy
    # at the equilibrium position, the energy should be zero
    pos = ho.positions.value_in_unit_system(unit.md_unit_system)
    e_chiron = harmonic_potential.compute_energy(pos)

    assert jnp.isclose(e_chiron, 0.0), "Energy at equilibrium position is not zero"
    pos = unit.Quantity(
        jnp.array(
            [
                [0.1, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [3.1, 0.0, 0.0],
                [4.1, 0.0, 0.0],
            ]
        ),
        unit.nanometer,
    )
    pos = pos.value_in_unit_system(unit.md_unit_system)

    e_chiron = harmonic_potential.compute_energy(pos)

    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"


def test_LJ_two_particle_system():
    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid
    from openmm import unit
    import jax.numpy as jnp

    # start with a 2 particle system
    from chiron.potential import LJPotential

    lj = LennardJonesFluid(reduced_density=1.5, nparticles=2)
    lj_pot = LJPotential(lj.topology)
    post = lj.positions.value_in_unit_system(unit.md_unit_system)
    e_chrion = lj_pot.compute_energy(post)

    # calculate the potential energy
    def calc_energy_for_2_particle_LJ(pos):
        # calculate energy
        sigma = unit.Quantity(3.350, unit.angstroms).value_in_unit_system(
            unit.md_unit_system
        )
        epsilon = unit.Quantity(1.0, unit.kilocalories_per_mole).value_in_unit_system(
            unit.md_unit_system
        )
        distances = jnp.linalg.norm(pos[0] - pos[1])
        return 4 * epsilon * ((sigma / distances) ** 12 - (sigma / distances) ** 6)

    e_ref = calc_energy_for_2_particle_LJ(post)
    assert jnp.isclose(e_chrion, e_ref), "LJ two particle energy is not correct"

    # compare forces
    import jax

    e_ref_force = -jax.grad(calc_energy_for_2_particle_LJ)(post)
    e_chrion_force = lj_pot.compute_force(post)
    assert jnp.allclose(
        e_chrion_force, e_ref_force
    ), "LJ two particle force is not correct"


def test_LJ_fluid():
    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid
    from openmm import unit
    import jax.numpy as jnp

    from chiron.potential import LJPotential
    from chiron.states import SamplerState

    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

    sigma = 0.34 * unit.nanometer  # argon
    epsilon = 0.238 * unit.kilocalories_per_mole
    cutoff = 3 * 0.34 * unit.nanometer
    skin = 0.5 * unit.nanometer

    for density in [0.5, 0.05, 0.005, 0.001]:
        lj_openmm = LennardJonesFluid(
            1000,
            reduced_density=density,
            sigma=sigma,
            epsilon=epsilon,
            cutoff=cutoff,
            switch_width=None,
            dispersion_correction=False,
            shift=False,
        )
        state = SamplerState(
            x0=lj_openmm.positions,
            box_vectors=lj_openmm.system.getDefaultPeriodicBoxVectors(),
        )

        nbr_list = NeighborListNsqrd(
            OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
        )
        nbr_list.build_from_state(state)

        lj_chiron = LJPotential(
            lj_openmm.topology, sigma=sigma, epsilon=epsilon, cutoff=cutoff
        )

        e_chiron_energy = lj_chiron.compute_energy(state.x0, nbr_list)
        e_openmm_energy = compute_openmm_reference_energy(
            lj_openmm, lj_openmm.positions
        )
        assert jnp.isclose(
            e_chiron_energy, e_openmm_energy.value_in_unit_system(unit.md_unit_system)
        ), "Chiron LJ fluid energy does not match openmm"


def test_ideal_gas(prep_temp_dir):
    from openmmtools.testsystems import IdealGas
    from openmm import unit

    # Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
    # For this example, the topology provides the masses for the particles
    # The default LennardJonesFluid example considers the system to be Argon with 39.9 amu
    n_particles = 216
    temperature = 298 * unit.kelvin
    pressure = 1 * unit.atmosphere
    mass = unit.Quantity(39.9, unit.gram / unit.mole)

    ideal_gas = IdealGas(
        nparticles=n_particles, temperature=temperature, pressure=pressure
    )

    from chiron.potential import IdealGasPotential
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

    # define the sampler state
    sampler_state = SamplerState(
        x0=ideal_gas.positions,
        box_vectors=ideal_gas.system.getDefaultPeriodicBoxVectors(),
    )

    from chiron.neighbors import PairList, OrthogonalPeriodicSpace

    # define the pair list for an orthogonal periodic space
    # since particles are non-interacting, this will not really do much
    # but will appropriately wrap particles in space
    nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)
    nbr_list.build_from_state(sampler_state)

    from chiron.reporters import SimulationReporter

    # initialize a reporter to save the simulation data
    filename = f"{prep_temp_dir}/test_ideal_gas_vol.h5"

    reporter1 = SimulationReporter(filename, ideal_gas.topology, 100)

    from chiron.mcmc import (
        MetropolisDisplacementMove,
        MoveSet,
        MCMCSampler,
        MCBarostatMove,
    )

    mc_disp_move = MetropolisDisplacementMove(
        seed=1234,
        displacement_sigma=0.1 * unit.nanometer,
        nr_of_moves=10,
    )

    mc_barostat_move = MCBarostatMove(
        seed=1234,
        volume_max_scale=0.2,
        nr_of_moves=100,
        adjust_box_scaling=True,
        adjust_frequency=50,
        simulation_reporter=reporter1,
    )
    move_set = MoveSet(
        [
            ("MetropolisDisplacementMove", mc_disp_move),
            ("MCBarostatMove", mc_barostat_move),
        ]
    )

    sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)
    sampler.run(n_iterations=30, nbr_list=nbr_list)  # how many times to repeat

    import h5py

    with h5py.File(filename, "r") as f:
        volume = f["volume"][:]
        steps = f["step"][:]

    # get expectations
    ideal_volume = ideal_gas.get_volume_expectation(thermodynamic_state)
    ideal_volume_std = ideal_gas.get_volume_standard_deviation(thermodynamic_state)

    volume_mean = jnp.mean(jnp.array(volume)) * unit.nanometer**3
    volume_std = jnp.std(jnp.array(volume)) * unit.nanometer**3

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
