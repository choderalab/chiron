def test_initialize_state():
    """Test initializing a thermodynamic and sampler state."""
    from chiron.states import ThermodynamicState, SamplerState
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from openmmtools.testsystems import HarmonicOscillator
    import jax.numpy as jnp

    ho = HarmonicOscillator()

    potential = HarmonicOscillatorPotential(ho.topology, k=ho.K, U0=ho.U0)

    thermodynamic_state = ThermodynamicState(potential)
    assert thermodynamic_state.temperature is None
    assert thermodynamic_state.pressure is None
    assert thermodynamic_state.volume is None
    assert thermodynamic_state.nr_of_particles is not None
    assert thermodynamic_state.potential is not None

    state = ThermodynamicState(
        potential, temperature=300 * unit.kelvin, volume=30 * (unit.angstrom**3)
    )
    assert state.temperature == 300 * unit.kelvin
    assert state.pressure is None
    assert state.volume == 30 * (unit.angstrom**3)
    assert state.nr_of_particles == 1
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())

    assert jnp.allclose(
        sampler_state.positions,
        jnp.array([[0.0, 0.0, 0.0]]),
    )


def test_sampler_state_conversion():
    """Test converting a sampler state to jnp arrays.
    Note, testing the conversion of positions, where internal unit length is nanometers
    and thus output jnp.arrays (with units dropped) should reflect this.
    """
    from chiron.states import SamplerState
    from openmm import unit
    import jax.numpy as jnp
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    sampler_state = SamplerState(
        unit.Quantity(jnp.array([[10.0, 10.0, 10.0]]), unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
    )

    assert jnp.allclose(
        sampler_state.positions,
        jnp.array([[10.0, 10.0, 10.0]]),
    )

    sampler_state = SamplerState(
        unit.Quantity(jnp.array([[10.0, 10.0, 10.0]]), unit.angstrom),
        current_PRNG_key=PRNG.get_random_key(),
    )

    assert jnp.allclose(
        sampler_state.positions,
        jnp.array([[1.0, 1.0, 1.0]]),
    )


def test_sampler_state_inputs():
    from chiron.states import SamplerState
    from openmm import unit
    import jax.numpy as jnp
    import pytest
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    # test input of positions
    # should have units
    with pytest.raises(TypeError):
        SamplerState(positions=jnp.array([1, 2, 3]))
    # throw and error because of incompatible units
    with pytest.raises(ValueError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.radians),
            current_PRNG_key=PRNG.get_random_key(),
        )

    # test input of velocities
    # velocities should have units
    with pytest.raises(TypeError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            velocities=jnp.array([1, 2, 3]),
        )
    # velocities should have units of distance/time
    with pytest.raises(ValueError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            velocities=unit.Quantity(jnp.array([1, 2, 3]), unit.nanometers),
        )

    # test input of box_vectors
    # box_vectors should have units
    with pytest.raises(TypeError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            box_vectors=jnp.array([1, 2, 3]),
        )
    # box_vectors should have units of distance
    with pytest.raises(ValueError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            box_vectors=unit.Quantity(
                jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), unit.radians
            ),
        )
    # check to see that the size of the box vectors are correct
    with pytest.raises(ValueError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            box_vectors=unit.Quantity(
                jnp.array([[1, 0, 0], [0, 1, 0]]), unit.nanometers
            ),
        )

    openmm_box = [
        unit.Quantity([4.031145745088737, 0.0, 0.0], unit.nanometer),
        unit.Quantity([0.0, 4.031145745088737, 0.0], unit.nanometer),
        unit.Quantity([0.0, 0.0, 4.031145745088737], unit.nanometer),
    ]

    # check openmm_box conversion:
    state = SamplerState(
        positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=openmm_box,
    )
    assert jnp.allclose(
        state.box_vectors,
        jnp.array(
            [[4.0311456, 0.0, 0.0], [0.0, 4.0311456, 0.0], [0.0, 0.0, 4.0311456]]
        ),
        atol=1e-4,
    )

    # openmm box vectors end up as a list with contents; check to make sure we capture an error if we pass a bad list
    with pytest.raises(TypeError):
        SamplerState(
            positions=unit.Quantity(jnp.array([[1, 2, 3]]), unit.nanometers),
            current_PRNG_key=PRNG.get_random_key(),
            box_vectors=[123],
        )


def test_reduced_potential():
    """Test the reduced potential function."""
    from chiron.states import ThermodynamicState, SamplerState
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.utils import PRNG

    ho = HarmonicOscillator()
    potential = HarmonicOscillatorPotential(topology=ho.topology, k=ho.K, U0=ho.U0)

    state = ThermodynamicState(
        potential, temperature=300 * unit.kelvin, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions, current_PRNG_key=PRNG.get_random_key())

    reduced_e = state.get_reduced_potential(sampler_state)
    assert reduced_e == 0.0
