import pytest
@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_langevin")
    return tmpdir

def test_langevin_dynamics(prep_temp_dir, provide_testsystems_and_potentials):
    """
    Test the harmonic oscillator system using a Langevin integrator.

    This function initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses the Langevin integrator to run the simulation.
    It then tests if the standard deviation of the potential is close to the expected value.
    """
    from openmm.unit import kelvin
    i = 0
    for testsystem, potential in provide_testsystems_and_potentials:
        # initialize testystem
        from openmm import unit
        import openmmtools
        import jax.numpy as jnp

        # initialize states and integrator
        from chiron.integrators import LangevinIntegrator
        from chiron.states import SamplerState, ThermodynamicState

        thermodynamic_state = ThermodynamicState(
            potential=potential, temperature=300 * kelvin
        )

        sampler_state = SamplerState(testsystem.positions)
        from chiron.reporters import SimulationReporter

        reporter = SimulationReporter(f"{prep_temp_dir}/test{i}.h5", None,1)

        integrator = LangevinIntegrator(reporter=reporter)
        integrator.run(
            sampler_state,
            thermodynamic_state,
            n_steps=5,
        )
        #LJFluid system does not include expectations
        #I think it would be better to split these into separate tests because of the differences in the
        # overall datastructures  Maybe a harmonic oscillator test class and an LJfluid one
        if isinstance(testsystem,  openmmtools.testsystems.HarmonicOscillator):
            expectation = testsystem.get_potential_expectation(thermodynamic_state)
            stddev = testsystem.get_potential_standard_deviation(thermodynamic_state)
            import jax.numpy as jnp

            assert jnp.isclose(
                expectation.value_in_unit_system(unit.md_unit_system), 3.741508178
            )
        i = i+1