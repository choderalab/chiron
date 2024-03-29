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
        from chiron.utils import PRNG

        PRNG.set_seed(1234)

        thermodynamic_state = ThermodynamicState(
            potential=potential, temperature=300 * unit.kelvin
        )

        sampler_state = SamplerState(testsystem.positions, PRNG.get_random_key())

        from chiron.reporters import LangevinDynamicsReporter
        from chiron.reporters import BaseReporter

        # set up reporter directory
        BaseReporter.set_directory(prep_temp_dir.join(f"test_{i}"))

        reporter = LangevinDynamicsReporter()

        integrator = LangevinIntegrator(
            reporter=reporter, report_interval=1, refresh_velocities=True
        )
        updated_sampler_state, updated_nbr_list = integrator.run(
            sampler_state,
            thermodynamic_state,
            number_of_steps=20,
        )
        i = i + 1
