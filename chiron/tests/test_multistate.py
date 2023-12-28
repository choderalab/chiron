from chiron.multistate import MultiStateSampler
import pytest


@pytest.fixture
def ho_multistate_sampler() -> MultiStateSampler:
    """
    Create a MultiStateSampler object for performing multistate simulations for a harmonic oscillator.

    Returns:
        MultiStateSampler: The created MultiStateSampler object.
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

    ho_potential = HarmonicOscillatorPotential(ho.topology)
    thermodynamic_states = [
        ThermodynamicState(ho_potential, temperature=T) for T in temperatures
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
    multistate_sampler = MultiStateSampler(mcmc_moves=move, number_of_iterations=2)
    multistate_sampler.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=sampler_state,
        nbr_list=nbr_list,
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
    ho_multistate_sampler.minimize()
