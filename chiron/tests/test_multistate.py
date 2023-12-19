def test_HO():
    import math
    from openmm import unit
    from openmmtools import testsystems
    from chiron.multistate import MultiStateSampler
    from chiron.mcmc import LangevinDynamicsMove
    from chiron.states import ThermodynamicState

    testsystem = testsystems.AlanineDipeptideImplicit()

    n_replicas = 3
    T_min = 298.0 * unit.kelvin  # Minimum temperature.
    T_max = 600.0 * unit.kelvin  # Maximum temperature.
    temperatures = [
        T_min
        + (T_max - T_min)
        * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
        / (math.e - 1.0)
        for i in range(n_replicas)
    ]
    temperatures = [
        T_min
        + (T_max - T_min)
        * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
        / (math.e - 1.0)
        for i in range(n_replicas)
    ]
    thermodynamic_states = [
        ThermodynamicState(system=testsystem.system, temperature=T)
        for T in temperatures
    ]

    # Initialize simulation object with options. Run with a langevin integrator.

    move = LangevinDynamicsMove(timestep=2.0 * unit.femtoseconds, n_steps=50)
    simulation = MultiStateSampler(mcmc_moves=move, number_of_iterations=2)

    # Run the simulation
    simulation.run()
