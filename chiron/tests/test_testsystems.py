def test_HO():
    """
    Test the harmonic oscillator system using a Langevin integrator.

    This function initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses the Langevin integrator to run the simulation.
    """
    from openmm.unit import kelvin

    # initialize testystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)
    # initialize states
    from chiron.states import SamplerState, ThermodynamicState

    thermodynamic_state = ThermodynamicState(
        potential=harmonic_potential, temperature=300 * kelvin
    )

    # don't do anything, just check that we get the right expectation value for the testsystem
    # NOTE: this doesn't actually test anything meaningful, but it's a good starting point
    expectation = ho.get_potential_standard_deviation(thermodynamic_state)
    from openmm import unit
    import jax.numpy as jnp

    assert jnp.isclose(
        expectation.value_in_unit_system(unit.md_unit_system), 3.741508178
    )
