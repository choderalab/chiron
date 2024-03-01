import copy

from chiron.multistate import MultiStateSampler
from chiron.neighbors import PairListNsqrd
import pytest
from typing import Tuple


def setup_sampler() -> Tuple[PairListNsqrd, MultiStateSampler]:
    """
    Set up the pair list and multistate sampler for the simulation.

    Returns:
        Tuple: A tuple containing the pair list and multistate sampler objects.
    """
    from openmm import unit
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.neighbors import PairListNsqrd, OrthogonalNonPeriodicSpace
    from chiron.reporters import MultistateReporter, BaseReporter
    from chiron.mcmc import MCMCSampler, MoveSchedule

    cutoff = 1.0 * unit.nanometer

    nbr_list = PairListNsqrd(OrthogonalNonPeriodicSpace(), cutoff=cutoff)

    lang_move = LangevinDynamicsMove(
        timestep=1.0 * unit.femtoseconds, number_of_steps=100
    )
    BaseReporter.set_directory("multistate_test")
    reporter = MultistateReporter()
    reporter.reset_reporter_file()
    move_schedule = MoveSchedule([("LangevinDynamicsMove", lang_move)])
    mcmc_sampler = MCMCSampler(
        move_schedule,
    )

    multistate_sampler = MultiStateSampler(mcmc_sampler=mcmc_sampler, reporter=reporter)
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
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = [SamplerState(ho.positions, PRNG.get_random_key()) for _ in x0s]
    nbr_list, multistate_sampler = setup_sampler()
    import copy

    nbr_lists = [copy.deepcopy(nbr_list) for _ in x0s]

    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_lists=nbr_lists,
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
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = [
        SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())
        for _ in sigmas
    ]
    import numpy as np

    f_i = np.array(
        [
            -np.log(2 * np.pi * (sigma / unit.angstroms) ** 2) * (3.0 / 2.0)
            for sigma in sigmas
        ]
    )

    nbr_list, multistate_sampler = setup_sampler()
    import copy

    nbr_lists = [copy.deepcopy(nbr_list) for _ in sigmas]

    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_lists=nbr_lists,
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
        ho_multistate_sampler_multiple_minima.sampler_states[0].positions,
        np.array([[0.0, 0.0, 0.0]]),
    )
    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[1].positions,
        np.array([[0.05, 0.0, 0.0]]),
        atol=1e-2,
    )
    assert np.allclose(
        ho_multistate_sampler_multiple_minima.sampler_states[2].positions,
        np.array([[0.1, 0.0, 0.0]]),
        atol=1e-2,
    )


# @pytest.mark.skip(
#     reason="Multistate code still needs to be modified in the multistage branch"
# )
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

    n_iterations = 25
    ho_sampler.run(n_iterations)

    # check that we have the correct number of iterations, replicas and states
    assert ho_sampler.iteration == n_iterations
    assert ho_sampler._iteration == n_iterations
    assert ho_sampler.n_replicas == 4
    assert ho_sampler.n_states == 4

    u_kn = ho_sampler._reporter.get_property("u_kn")

    # the u_kn array  is transposed to be _states, n_replicas, n_iterations
    # SHOULD THIS BE TRANSPOSED IN THE REPORTER? I feel safer to have it
    # be transposed when used (if we want it in such a form).
    # note n_iterations+1 because it logs time = 0 as well
    assert u_kn.shape == (4, 4, n_iterations + 1)
    # check that the free energies are correct
    print(ho_sampler.analytical_f_i)
    # [ 0.        , -0.28593054, -0.54696467, -0.78709279]
    print(ho_sampler.delta_f_ij_analytical)
    print(ho_sampler.f_k)

    assert np.allclose(ho_sampler.delta_f_ij_analytical[0], ho_sampler.f_k, atol=0.1)
