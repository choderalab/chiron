def test_initialize_state():
    from chiron.states import SimulationState
    from openmm import unit

    state = SimulationState()
    assert state.temperature is None
    assert state.pressure is None
    assert state.volume is None
    assert state.nr_of_particles is None
    assert state.position is None
    assert state.potential is None

    state = SimulationState(
        temperature=300, volume=30 * (unit.angstrom**3), nr_of_particles=3000
    )
    assert state.temperature == 300
    assert state.pressure is None
    assert state.volume == 30 * (unit.angstrom**3)
    assert state.nr_of_particles == 3000
    assert state.position is None


def test_reduced_potential():
    from chiron.states import SimulationState
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmmtools.testsystems import HarmonicOscillator

    state = SimulationState(
        temperature=300, volume=30 * (unit.angstrom**3), nr_of_particles=1
    )
    ho = HarmonicOscillator()

    harmonic_potential = HarmonicOscillatorPotential(
        ho.K, jnp.array([0, 0, 0]) * unit.angstrom, 0.0
    )
    state.potential = harmonic_potential
    assert state.reduced_potential() == 0.5
