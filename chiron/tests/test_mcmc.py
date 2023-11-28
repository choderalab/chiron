def test_sample_from_harmonic_osciallator():
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.integrator import LangevinIntegrator

    ho = HarmonicOscillator()
    harmonic_potential = HarmonicOscillatorPotential(ho.K, 0.0 * unit.angstrom, ho.U0)
    integrator = LangevinIntegrator(harmonic_potential, ho.topology)
    r = integrator.run(
        ho.positions,
        temperature=300 * unit.kelvin,
        n_steps=5,
        stepsize=0.2 * unit.femtosecond,
    )

    reference_energy = jnp.array(
        [0.0, 0.00018982, 0.00076115, 0.00172312, 0.00307456, 0.00480607]
    )

    jnp.allclose(jnp.array(r["energy"]).flatten(), reference_energy)


def test_sample_from_joint_distribution_of_two_HO():
    pass


def test_sample_from_joint_distribution_of_two_HO_with_local_moves():
    pass
