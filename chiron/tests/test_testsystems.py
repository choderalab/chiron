from chiron.integrator import LangevinIntegrator


def test_HO():
    from openmm.unit import kelvin
    from chiron.potential import HarmonicOscillatorPotential
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    harmonic_potential = HarmonicOscillatorPotential(ho.K, ho.positions, ho.U0)
    integrator = LangevinIntegrator(harmonic_potential, ho.topology)
    integrator.run(ho.positions, temperature=300 * kelvin, n_steps=1000)
