from chiron.integrators import LangevinIntegrator


def test_HO():
    from openmm.unit import kelvin
    from chiron.potential import HarmonicOscillatorPotential
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology, ho.K, ho.positions, ho.U0
    )
    integrator = LangevinIntegrator()
    integrator.run(
        ho.positions,
        harmonic_potential,
        temperature=300 * kelvin,
        n_steps=5,
    )

    class State:
        def __init__(self, temperature):
            self.temperature = temperature

    stddev = ho.get_potential_expectation(State(300 * kelvin))
    expectation = ho.get_potential_standard_deviation(State(300 * kelvin))
    from openmm import unit
    import jax.numpy as jnp

    assert jnp.isclose(
        expectation.value_in_unit_system(unit.md_unit_system), 3.741508178
    )
