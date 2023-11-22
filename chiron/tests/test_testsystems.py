from chiron.testsystems import HarmonicOscillator
from chiron.integrator import LangevinIntegrator


def test_HO():
    ho = HarmonicOscillator()
    integrator = LangevinIntegrator(ho.harmonic_potential)
