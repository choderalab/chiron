def compute_openmm_reference_energy(testsystem, positions):
    from openmm import unit
    from openmm.app import Simulation
    from openmm import LangevinIntegrator

    system = testsystem.system
    integrator = LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.1 * unit.femtosecond
    )
    sim = Simulation(testsystem.topology, system, integrator)
    sim.context.setPositions(positions)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(e)
    return e


def test_HO():
    """
    Test the harmonic oscillator system using a Langevin integrator.

    This function initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses the Langevin integrator to run the simulation.
    """
    # initialize testystem
    from openmmtools.testsystems import HarmonicOscillator, HarmonicOscillatorArray

    ho = HarmonicOscillator()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    import jax.numpy as jnp
    from openmm import unit

    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology,
        ho.K,
        x0=unit.Quantity(jnp.array([[0.0, 0.0, 0.0]]), unit.angstrom),
        U0=ho.U0,
    )
    ###############################
    # Harmonic Oscillator
    ###############################
    # calculate the energy
    # at the equilibrium position, the energy should be zero
    pos = jnp.array([[0.0, 0.0, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    assert jnp.isclose(e_chiron, 0.0), "Energy at equilibrium position is not zero"

    # calculate the energy
    # off-equilibrium
    pos = jnp.array([[0.1, 0.0, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    # compare with openmm reference for same system
    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"

    pos = jnp.array([[0.0, 0.1, 0.0]])
    e_chiron = harmonic_potential.compute_energy(pos)
    # compare with openmm reference for same system
    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"

    ###############################
    # Harmonic OscillatorArray
    ###############################
    ho = HarmonicOscillatorArray()
    # initialize potential
    from chiron.potential import HarmonicOscillatorPotential

    import jax.numpy as jnp
    from openmm import unit

    harmonic_potential = HarmonicOscillatorPotential(
        ho.topology,
        ho.K,
        x0=unit.Quantity(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
            unit.nanometer,
        ),
    )

    # calculate the energy
    # at the equilibrium position, the energy should be zero
    pos = ho.positions.value_in_unit_system(unit.md_unit_system)
    e_chiron = harmonic_potential.compute_energy(pos)

    assert jnp.isclose(e_chiron, 0.0), "Energy at equilibrium position is not zero"
    pos = unit.Quantity(
        jnp.array(
            [
                [0.1, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [3.1, 0.0, 0.0],
                [4.1, 0.0, 0.0],
            ]
        ),
        unit.nanometer,
    )
    pos = pos.value_in_unit_system(unit.md_unit_system)

    e_chiron = harmonic_potential.compute_energy(pos)

    e_ref = compute_openmm_reference_energy(ho, pos)
    e_ref = e_ref.value_in_unit_system(unit.md_unit_system)
    assert jnp.isclose(e_chiron, e_ref), "Energy at equilibrium position is not zero"


def test_LJ_two_particle_system():
    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid
    from openmm import unit
    import jax.numpy as jnp

    # start with a 2 particle system
    from chiron.potential import LJPotential

    lj = LennardJonesFluid(reduced_density=1.5, nparticles=2)
    lj_pot = LJPotential()
    post = lj.positions.value_in_unit_system(unit.md_unit_system)
    e_chrion = lj_pot.compute_energy(post)

    # calculate the potential energy
    pos = jnp.array([[0.0, 0.0, 0.0], [0.18710922, 0.18710922, 0.18710922]])

    def calc_energy_for_2_particle_LJ(pos):
        # calculate energy
        sigma = unit.Quantity(3.350, unit.angstroms).value_in_unit_system(
            unit.md_unit_system
        )
        epsilon = unit.Quantity(1.0, unit.kilocalories_per_mole).value_in_unit_system(
            unit.md_unit_system
        )
        distances = jnp.linalg.norm(pos[0] - pos[1])
        return 4 * epsilon * ((sigma / distances) ** 12 - (sigma / distances) ** 6)

    e_ref = calc_energy_for_2_particle_LJ(pos)
    assert jnp.isclose(e_chrion, e_ref), "LJ two particle energy is not correct"

    # compare forces
    import jax

    e_ref_force = -jax.grad(calc_energy_for_2_particle_LJ)(pos)
    e_chrion_force = lj_pot.compute_force(post)
    assert jnp.allclose(
        e_chrion_force, e_ref_force
    ), "LJ two particle force is not correct"
    

def test_LJ_fluid():
    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid
    from openmm import unit
    import jax.numpy as jnp

    # start with a 2 particle system
    from chiron.potential import LJPotential

    lj = LennardJonesFluid(reduced_density=1.5, nparticles=200)
    lj_pot = LJPotential()
    post = lj.positions.value_in_unit_system(unit.md_unit_system)
    e_chrion = lj_pot.compute_energy(post)

    # calculate the potential energy
    pos = jnp.array([[0.0, 0.0, 0.0], [0.18710922, 0.18710922, 0.18710922]])

    def calc_energy_for_2_particle_LJ(pos):
        # calculate energy
        sigma = unit.Quantity(3.350, unit.angstroms).value_in_unit_system(
            unit.md_unit_system
        )
        epsilon = unit.Quantity(1.0, unit.kilocalories_per_mole).value_in_unit_system(
            unit.md_unit_system
        )
        distances = jnp.linalg.norm(pos[0] - pos[1])
        return 4 * epsilon * ((sigma / distances) ** 12 - (sigma / distances) ** 6)

    e_ref = calc_energy_for_2_particle_LJ(pos)
    assert jnp.isclose(e_chrion, e_ref), "LJ two particle energy is not correct"

    # compare forces
    import jax

    e_ref_force = -jax.grad(calc_energy_for_2_particle_LJ)(pos)
    e_chrion_force = lj_pot.compute_force(post)
    assert jnp.allclose(
        e_chrion_force, e_ref_force
    ), "LJ two particle force is not correct"
