import pytest

def test_langevin_dynamics(prep_temp_dir, provide_testsystems_and_potentials):
    """
    Test the Langevin integrator with a set of test systems.

    This function initializes openmmtools.testsystems,
    sets up a potential, and uses the Langevin integrator to run the simulation.
    """
    from openmm.unit import kelvin

    i = 0
    for testsystem, potential in provide_testsystems_and_potentials:
        # initialize testystem

        # initialize states and integrator
        from chiron.integrators import LangevinIntegrator
        from chiron.states import SamplerState, ThermodynamicState

        thermodynamic_state = ThermodynamicState(
            potential=potential, temperature=300 * kelvin
        )

        sampler_state = SamplerState(testsystem.positions)
        from chiron.reporters import SimulationReporter

        reporter = SimulationReporter(f"{prep_temp_dir}/test{i}.h5", None, 1)

        integrator = LangevinIntegrator(reporter=reporter)
        integrator.run(
            sampler_state,
            thermodynamic_state,
            n_steps=5,
        )
        i = i + 1
