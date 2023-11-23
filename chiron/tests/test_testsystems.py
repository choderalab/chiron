from chiron.testsystems import HarmonicOscillator
from chiron.integrator import LangevinIntegrator


def test_HO():
    from openmm.unit import kelvin

    ho = HarmonicOscillator()
    integrator = LangevinIntegrator(ho.harmonic_potential, ho.topology)
    integrator.run(ho.x0, temperature=300 * kelvin, n_steps=1000)
