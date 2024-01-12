from chiron.multistate import MultiStateSampler
from chiron.neighbors import NeighborListNsqrd
import pytest
from typing import Tuple


def setup_sampler() -> Tuple[NeighborListNsqrd, MultiStateSampler]:
    """
    Set up the neighbor list and multistate sampler for the simulation.

    Returns:
        Tuple: A tuple containing the neighbor list and multistate sampler objects.
    """
    from openmm import unit
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace
    from chiron.reporters import MultistateReporter

    sigma = 0.34 * unit.nanometer
    cutoff = 3.0 * sigma
    skin = 0.5 * unit.nanometer

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
    )

    move = LangevinDynamicsMove(stepsize=2.0 * unit.femtoseconds, nr_of_steps=500)
    reporter = MultistateReporter()
    reporter.reset_reporter_file()

    multistate_sampler = MultiStateSampler(mcmc_moves=move, reporter=reporter)
    return nbr_list, multistate_sampler


@pytest.fixture
def ho_multistate_sampler_multiple_minima() -> MultiStateSampler:
    """
    Create a multi-state sampler for multiple harmonic oscillators with different minimum values.

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
    Create a multi-state sampler for a harmonic oscillator system with different spring constants.
    Returns
    -------
    MultiStateSampler
        The multi-state sampler object.
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
    """
    Test initialization for the MultiStateSampler class.

    Parameters:
    -------
    ho_multistate_sampler_multiple_minima: MultiStateSampler
        An instance of the MultiStateSampler class.
    Raises:
    -------
    AssertionError:
        If any of the assertions fail.

    """
    assert ho_multistate_sampler_multiple_minima._iteration == 0
    assert ho_multistate_sampler_multiple_minima.n_replicas == 3
    assert ho_multistate_sampler_multiple_minima.n_states == 3
    assert ho_multistate_sampler_multiple_minima._energy_thermodynamic_states.shape == (
        3,
        3,
    )
    assert ho_multistate_sampler_multiple_minima._n_proposed_matrix.shape == (3, 3)


def test_multistate_minimize(ho_multistate_sampler_multiple_minima: MultiStateSampler):
    """
    Test function for the `minimize` method of the `ho_multistate_sampler` object.
    Check if the sampler states are correctly minimized.

    Parameters
    ----------
    ho_multistate_sampler: MultiStateSampler
    """

    import numpy as np

    ho_multistate_sampler_multiple_minima.minimize()

    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[0].x0,
        np.array([[0.0, 0.0, 0.0]]),
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
    """
    Test function for running the multistate sampler.

    Parameters
    ----------
    ho_multistate_sampler_multiple_ks: MultiStateSampler
        The multistate sampler object.
    Raises
    -------
        AssertionError: If free energy does not converge to the analytical free energy difference.

    """

    ho_sampler = ho_multistate_sampler_multiple_ks
    import numpy as np

    print(f"Analytical free energy difference: {ho_sampler.delta_f_ij_analytical[0]}")

    n_iteratinos = 25
    ho_sampler.run(n_iteratinos)

    # check that we have the correct number of iterations, replicas and states
    assert ho_sampler.iteration == n_iteratinos
    assert ho_sampler._iteration == n_iteratinos
    assert ho_sampler.n_replicas == 4
    assert ho_sampler.n_states == 4

    # check that the free energies are correct
    print(ho_sampler.analytical_f_i)
    print(ho_sampler.delta_f_ij_analytical)
    print(ho_sampler.f_k)

    assert np.allclose(ho_sampler.delta_f_ij_analytical[0], ho_sampler.f_k, atol=0.1)
