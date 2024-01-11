import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_langevin")
    return tmpdir


def test_langevin_dynamics(prep_temp_dir, provide_testsystems_and_potentials):
    """
    Test the Langevin integrator with a set of test systems.

    This function initializes openmmtools.testsystems,
    sets up a potential, and uses the Langevin integrator to run the simulation.
    """
    from openmm import unit

    i = 0
    for testsystem, potential in provide_testsystems_and_potentials:
        # initialize testystem

        # initialize states and integrator
        from chiron.integrators import LangevinIntegrator
        from chiron.states import SamplerState, ThermodynamicState

        thermodynamic_state = ThermodynamicState(
            potential=potential, temperature=300 * unit.kelvin
        )

        sampler_state = SamplerState(testsystem.positions)
        from chiron.reporters import LangevinDynamicsReporter
        from chiron.reporters import BaseReporter

        BaseReporter.set_directory(prep_temp_dir.join(f"test_{i}"))
        reporter = LangevinDynamicsReporter()

        integrator = LangevinIntegrator(reporter=reporter, save_frequency=1)
        integrator.run(
            sampler_state,
            thermodynamic_state,
            n_steps=20,
        )
        i = i + 1
