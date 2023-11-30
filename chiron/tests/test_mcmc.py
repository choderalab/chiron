def test_sample_from_harmonic_osciallator():
    # use local moves to sample from the harmonic oscillator
    from openmm import unit

    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    params = {}
    params["k"] = ho.K
    params["U0"] = ho.U0
    params["x0"] = 0.0 * unit.angstrom
    params["topology"] = ho.topology

    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(None, **params)
    # NOTE: let's construct this potential from the openmmtools test system (ask dominic how to do this)
    from chiron.integrators import LangevinIntegrator

    integrator = LangevinIntegrator(
        stepsize=0.2 * unit.femtosecond,
    )

    r = integrator.run(
        ho.positions,
        harmonic_potential,
        temperature=300 * unit.kelvin,
        n_steps=5,
    )

    import jax.numpy as jnp

    reference_energy = jnp.array(
        [0.0, 0.00018982, 0.00076115, 0.00172312, 0.00307456, 0.00480607]
    )
    jnp.allclose(jnp.array(r["energy"]).flatten(), reference_energy)


def test_sample_from_harmonic_osciallator_with_MCMC_classes():
    # use local moves to sample from the HO, but use the MCMC classes
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.mcmc import LangevinDynamicsMove, MoveSet, GibbsSampler
    from chiron.states import SimulationState

    state = SimulationState()

    ho = HarmonicOscillator()

    params = {}
    params["k"] = ho.K
    params["U0"] = ho.U0
    params["x0"] = 0.0 * unit.angstrom
    params["topology"] = ho.topology

    harmonic_potential = HarmonicOscillatorPotential(None, **params)

    langevin_move = LangevinDynamicsMove(
        NeuralNetworkPotential=HarmonicOscillatorPotential,
        stepsize=0.2 * unit.femtoseconds,
    )

    move_set = MoveSet({"LangevinDynamics": langevin_move}, [("LangevinDynamics", 10)])
    sampler = GibbsSampler(state, move_set)
    sampler.run(ho.positions)


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
