def test_HO():
    from openmm.unit import kelvin

    # initialize testystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # initialize states and integrator
    from chiron.integrators import LangevinIntegrator
    from chiron.states import SamplerState, ThermodynamicState

    thermodynamic_state = ThermodynamicState(
        potential=harmonic_potential, temperature=300 * kelvin
    )
    sampler_state = SamplerState(ho.positions)
    integrator = LangevinIntegrator()
    integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=5,
    )

    stddev = ho.get_potential_expectation(thermodynamic_state)
    expectation = ho.get_potential_standard_deviation(thermodynamic_state)
    from openmm import unit
    import jax.numpy as jnp

    assert jnp.isclose(
        expectation.value_in_unit_system(unit.md_unit_system), 3.741508178
    )
