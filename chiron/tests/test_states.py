def test_initialize_state():
    """Test initializing a thermodynamic and sampler state."""
    from chiron.states import ThermodynamicState, SamplerState
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from openmmtools.testsystems import HarmonicOscillator
    import jax.numpy as jnp

    ho = HarmonicOscillator()

    potential = HarmonicOscillatorPotential(ho.topology, ho.K, ho.U0)

    thermodynamic_state = ThermodynamicState(potential)
    assert thermodynamic_state.temperature is None
    assert thermodynamic_state.pressure is None
    assert thermodynamic_state.volume is None
    assert thermodynamic_state.nr_of_particles is not None
    assert thermodynamic_state.potential is not None

    state = ThermodynamicState(
        potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    assert state.temperature == 300
    assert state.pressure is None
    assert state.volume == 30 * (unit.angstrom**3)
    assert state.nr_of_particles == 1

    sampler_state = SamplerState(ho.positions)

    assert jnp.allclose(
        sampler_state.x0,
        jnp.array([[0.0, 0.0, 0.0]]),
    )


def test_reduced_potential():
    """Test the reduced potential function."""
    from chiron.states import ThermodynamicState, SamplerState
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    potential = HarmonicOscillatorPotential(topology=ho.topology, k=ho.K, U0=ho.U0)

    state = ThermodynamicState(
        potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions)

    reduced_e = state.get_reduced_potential(sampler_state)
    assert reduced_e == 0.0
