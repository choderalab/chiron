from chiron.multistate import MultiStateSampler
import pytest


@pytest.fixture
def ho_multistate_sampler() -> MultiStateSampler:
    """
    Create a multi-state sampler for a harmonic oscillator system.

    Returns:
        MultiStateSampler: The multi-state sampler object.
    """
    import math
    from openmm import unit
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.states import ThermodynamicState, SamplerState
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

    ho = HarmonicOscillator()
    n_replicas = 3
    T_min = 298.0 * unit.kelvin  # Minimum temperature.
    T_max = 600.0 * unit.kelvin  # Maximum temperature.
    temperatures = [
        T_min
        + (T_max - T_min)
        * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
        / (math.e - 1.0)
        for i in range(n_replicas)
    ]
    import jax.numpy as jnp

    x0s = [
        unit.Quantity(jnp.array([[x0, 0.0, 0.0]]), unit.angstrom)
        for x0 in jnp.linspace(0.0, 1.0, n_replicas)
    ]
    thermodynamic_states = [
        ThermodynamicState(
            HarmonicOscillatorPotential(ho.topology, x0=x0), temperature=T
        )
        for T, x0 in zip(temperatures, x0s)
    ]
    sampler_state = [SamplerState(ho.positions) for _ in temperatures]

    # Initialize simulation object with options. Run with a langevin integrator.
    # initialize the LennardJones potential in chiron
    #
    sigma = 0.34 * unit.nanometer
    cutoff = 3.0 * sigma
    skin = 0.5 * unit.nanometer

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
    )

    move = LangevinDynamicsMove(stepsize=2.0 * unit.femtoseconds, nr_of_steps=50)

    from openmmtools.multistate import MultiStateReporter

    reporter = MultiStateReporter("test.nc")

    multistate_sampler = MultiStateSampler(mcmc_moves=move, number_of_iterations=2)
    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_list=nbr_list,
        reporter=reporter,
    )

    return multistate_sampler


def test_multistate_class(ho_multistate_sampler):
    # test the multistate_sampler object
    assert ho_multistate_sampler.number_of_iterations == 2
    assert ho_multistate_sampler.n_replicas == 3
    assert ho_multistate_sampler.n_states == 3
    assert ho_multistate_sampler._energy_thermodynamic_states.shape == (3, 3)
    assert ho_multistate_sampler._n_proposed_matrix.shape == (3, 3)


def test_multistate_minimize(ho_multistate_sampler):
    """
    Test function for the `minimize` method of the `ho_multistate_sampler` object.
    It checks if the sampler states are correctly minimized.

    Parameters
    ----------
    ho_multistate_sampler: The `ho_multistate_sampler` object to be tested.
    """

    import numpy as np

    ho_multistate_sampler.minimize()

    assert np.allclose(
        ho_multistate_sampler.sampler_states[0].x0, np.array([[0.0, 0.0, 0.0]])
    )
    assert np.allclose(
        ho_multistate_sampler.sampler_states[1].x0,
        np.array([[0.05, 0.0, 0.0]]),
        atol=1e-2,
    )
    assert np.allclose(
        ho_multistate_sampler.sampler_states[2].x0,
        np.array([[0.1, 0.0, 0.0]]),
        atol=1e-2,
    )


def test_multistate_equilibration(ho_multistate_sampler):
    import numpy as np

    ho_multistate_sampler.equilibrate(10)

    assert np.allclose(
        ho_multistate_sampler._replica_thermodynamic_states, np.array([0, 1, 2])
    )
    assert np.allclose(
        ho_multistate_sampler._energy_thermodynamic_states,
        np.array(
            [
                [4.81132936, 3.84872651, 3.10585403],
                [6.54490519, 5.0176239, 3.85019779],
                [9.48260307, 7.07196712, 5.21255827],
            ]
        ),
    )


@pytest.fixture
def ho_multistate_sampler_single_sampler_state() -> MultiStateSampler:
    """
    Create a multi-state sampler for a harmonic oscillator system.

    Returns:
        MultiStateSampler: The multi-state sampler object.
    """
    from openmm import unit
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.states import ThermodynamicState, SamplerState
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

    ho = HarmonicOscillator()
    n_states = 4
    
    T = 300.0 * unit.kelvin  # Minimum temperature.
    kT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA
    sigmas = [
        unit.Quantity(1.0 + 0.2 * state_index, unit.angstrom)
        for state_index in range(n_states)
    ]
    Ks = [kT / sigma ** 2 for sigma in sigmas]
    thermodynamic_states = [
        ThermodynamicState(
            HarmonicOscillatorPotential(ho.topology, k=k), temperature=T
        )
        for k in Ks
    ]
    sampler_state = [SamplerState(ho.positions) for _ in sigmas]
    import numpy as np

    f_i = np.array(
        [
            -np.log(2 * np.pi * (sigma / unit.angstroms) ** 2) * (3.0 / 2.0)
            for sigma in sigmas
        ]
    )

    # Initialize simulation object with options. Run with a langevin integrator.
    # initialize the LennardJones potential in chiron
    #
    sigma = 0.34 * unit.nanometer
    cutoff = 3.0 * sigma
    skin = 0.5 * unit.nanometer

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=10
    )

    move = LangevinDynamicsMove(stepsize=1.0 * unit.femtoseconds, nr_of_steps=100)

    from openmmtools.multistate import MultiStateReporter

    reporter = MultiStateReporter("test.nc")

    multistate_sampler = MultiStateSampler(mcmc_moves=move, number_of_iterations=10)
    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_list=nbr_list,
        reporter=reporter,
    )
    multistate_sampler.analytical_f_i = f_i
    multistate_sampler.delta_f_ij_analytical = f_i - f_i[:, np.newaxis]
    return multistate_sampler


def test_multistate_run(ho_multistate_sampler_single_sampler_state):
    ho_sampler = ho_multistate_sampler_single_sampler_state
    import numpy as np

    ho_sampler.equilibrate(10)
    ho_sampler.run(200)
    print(ho_sampler.analytical_f_i)
    print(ho_sampler.delta_f_ij_analytical)
    a = 7
