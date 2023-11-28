def test_sample_from_harmonic_osciallator():
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    import jax.numpy as jnp
    from openmmtools.testsystems import HarmonicOscillator
    from chiron.integrator import LangevinIntegrator  

    ho = HarmonicOscillator()
    harmonic_potential = HarmonicOscillatorPotential(
        ho.K, 0.0 * unit.angstrom, ho.U0
    )
    integrator = LangevinIntegrator(harmonic_potential, ho.topology)
    r = integrator.run(ho.positions, temperature=300 * unit.kelvin, n_steps=5000, stepsize=0.5 * unit.femtosecond)




def test_sample_from_joint_distribution_of_two_HO():
    pass

def test_sample_from_joint_distribution_of_two_HO_with_local_moves():
    pass