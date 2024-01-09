from chiron.multistate import MultiStateSampler
import pytest


def setup_sampler():
    from openmm import unit
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

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

    multistate_sampler = MultiStateSampler(mcmc_moves=move)
    return nbr_list, multistate_sampler


@pytest.fixture
def ho_multistate_sampler_multiple_minima() -> MultiStateSampler:
    """
    Create a multi-state sampler for a harmonic oscillator system.

    Returns:
        MultiStateSampler: The multi-state sampler object.
    """
    from chiron.states import ThermodynamicState, SamplerState
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmm import unit

    n_replicas = 3
    T = 300.0 * unit.kelvin  # Minimum temperature.
    x0s = [
        unit.Quantity(jnp.array([[x0, 0.0, 0.0]]), unit.angstrom)
        for x0 in jnp.linspace(0.0, 1.0, n_replicas)
    ]

    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    thermodynamic_states = [
        ThermodynamicState(
            HarmonicOscillatorPotential(ho.topology, x0=x0), temperature=T
        )
        for x0 in x0s
    ]
    sampler_state = [SamplerState(ho.positions) for _ in x0s]
    nbr_list, multistate_sampler = setup_sampler()
    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_list=nbr_list,
    )

    return multistate_sampler


@pytest.fixture
def ho_multistate_sampler_multiple_ks() -> MultiStateSampler:
    """
    Create a multi-state sampler for a harmonic oscillator system.

    Returns:
        MultiStateSampler: The multi-state sampler object.
    """
    from openmm import unit
    from chiron.states import ThermodynamicState, SamplerState
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.potential import HarmonicOscillatorPotential

    ho = HarmonicOscillator()
    n_states = 4

    T = 300.0 * unit.kelvin  # Minimum temperature.
    kT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA
    sigmas = [
        unit.Quantity(2.0 + 0.2 * state_index, unit.angstrom)
        for state_index in range(n_states)
    ]
    Ks = [kT / sigma**2 for sigma in sigmas]

    thermodynamic_states = [
        ThermodynamicState(HarmonicOscillatorPotential(ho.topology, k=k), temperature=T)
        for k in Ks
    ]
    from loguru import logger as log

    log.info(f"Initialize harmonic oscillator with {n_states} states and ks {Ks}")

    sampler_state = [SamplerState(ho.positions) for _ in sigmas]
    import numpy as np

    f_i = np.array(
        [
            -np.log(2 * np.pi * (sigma / unit.angstroms) ** 2) * (3.0 / 2.0)
            for sigma in sigmas
        ]
    )

    nbr_list, multistate_sampler = setup_sampler()

    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_list=nbr_list,
    )
    multistate_sampler.analytical_f_i = f_i
    multistate_sampler.delta_f_ij_analytical = f_i - f_i[:, np.newaxis]
    return multistate_sampler


def test_multistate_class(ho_multistate_sampler_multiple_minima: MultiStateSampler):
    # test the multistate_sampler object
    assert ho_multistate_sampler_multiple_minima._iteration == 0
    assert ho_multistate_sampler_multiple_minima.n_replicas == 3
    assert ho_multistate_sampler_multiple_minima.n_states == 3
    assert ho_multistate_sampler_multiple_minima._energy_thermodynamic_states.shape == (3, 3)
    assert ho_multistate_sampler_multiple_minima._n_proposed_matrix.shape == (3, 3)


def test_multistate_minimize(ho_multistate_sampler_multiple_minima: MultiStateSampler):
    """
    Test function for the `minimize` method of the `ho_multistate_sampler` object.
    It checks if the sampler states are correctly minimized.

    Parameters
    ----------
    ho_multistate_sampler: The `ho_multistate_sampler` object to be tested.
    """

    import numpy as np

    ho_multistate_sampler_multiple_minima.minimize()

    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[0].x0, np.array([[0.0, 0.0, 0.0]])
    )
    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[1].x0,
        np.array([[0.05, 0.0, 0.0]]),
        atol=1e-2,
    )
    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[2].x0,
        np.array([[0.1, 0.0, 0.0]]),
        atol=1e-2,
    )


def test_multistate_run(ho_multistate_sampler_multiple_ks: MultiStateSampler):
    ho_sampler = ho_multistate_sampler_multiple_ks
    import numpy as np

    n_iteratinos = 100
    ho_sampler.run(n_iteratinos)

    # check that we have the correct number of iterations, replicas and states
    assert ho_sampler.iteration == n_iteratinos
    assert ho_sampler._iteration == n_iteratinos
    assert ho_sampler.n_replicas == 4
    assert ho_sampler.n_states == 4

    # check that the free energies are correct
    print(ho_sampler.analytical_f_i)
    print(ho_sampler.delta_f_ij_analytical)
    print(ho_sampler._last_mbar_f_k_offline)
    a = 7
